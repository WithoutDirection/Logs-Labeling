#!/usr/bin/env python3
"""
Embedding quality benchmark on triplet datasets.

Goal:
	Given triplets (Base, Near, Far), evaluate whether an embedding model
	correctly scores Near closer to Base than Far.

Default dataset:
	data/testingdataset/*.csv

Outputs:
	result/embedding_model_benchmark/summary.csv
	result/embedding_model_benchmark/model_ranking.csv
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import re

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Import project modules
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PKG_DIR = REPO_ROOT / "Logs Labeling"
if str(PKG_DIR) not in sys.path:
	sys.path.insert(0, str(PKG_DIR))

_config_spec = importlib.util.spec_from_file_location("project_config", PKG_DIR / "config.py")
if _config_spec is None or _config_spec.loader is None:
	raise ImportError(f"Cannot load config.py from {PKG_DIR}")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

_bert_spec = importlib.util.spec_from_file_location("project_models_bert", PKG_DIR / "models" / "bert.py")
if _bert_spec is None or _bert_spec.loader is None:
	raise ImportError(f"Cannot load models/bert.py from {PKG_DIR}")
_bert_mod = importlib.util.module_from_spec(_bert_spec)
_bert_spec.loader.exec_module(_bert_mod)

get_bert_model = _bert_mod.get_bert_model
list_available_models = _bert_mod.list_available_models


@dataclass
class EvalResult:
	model: str
	preprocess: str
	dataset: str
	n_rows: int
	accuracy: float
	ties: int
	mean_margin: float
	median_margin: float
	p10_margin: float


def _parse_dataset_meta(dataset_name: str) -> tuple[str, str]:
	"""Parse dataset filename into (category, strategy)."""
	name = dataset_name.lower()
	if "file" in name:
		category = "File"
	elif "registry" in name:
		category = "Registry"
	elif "network" in name:
		category = "Network"
	else:
		category = "Unknown"

	m = re.search(r"strategy_([a-z0-9]+)", name)
	strategy = m.group(1).upper() if m else "Unknown"
	return category, strategy


def _resolve_models(models_arg: str) -> List[str]:
	available = list(list_available_models().keys())
	if models_arg.strip().lower() == "all":
		return available

	requested = [m.strip() for m in models_arg.split(",") if m.strip()]
	unknown = [m for m in requested if m not in available]
	if unknown:
		raise ValueError(f"Unknown model key(s): {unknown}. Available: {available}")
	return requested


def _resolve_preprocesses(pre_arg: str) -> List[str]:
	valid = {"raw", "none", "drain", "ants", "spell", "logmine", "lke"}
	if pre_arg.strip().lower() == "all":
		return ["raw", "drain", "ants", "spell", "logmine", "lke"]

	requested = [p.strip().lower() for p in pre_arg.split(",") if p.strip()]
	for p in requested:
		if p not in valid:
			raise ValueError(
				f"Unknown preprocess mode: {p}. Valid: raw, none, drain, ants, spell, logmine, lke, all"
			)
	return ["raw" if p == "none" else p for p in requested]


def _iter_csvs(dataset_dir: Path) -> Iterable[Path]:
	return sorted(p for p in dataset_dir.glob("*.csv") if p.is_file())


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""Row-wise cosine similarity."""
	a_norm = np.linalg.norm(a, axis=1)
	b_norm = np.linalg.norm(b, axis=1)
	denom = np.clip(a_norm * b_norm, 1e-12, None)
	return np.sum(a * b, axis=1) / denom


def _template_with_logpai(texts: List[str], parser_name: str) -> List[str]:
	"""Run LogPai-style parser (Spell/LogMine/LKE) on an in-memory text list."""
	with tempfile.TemporaryDirectory(prefix=f"pre_{parser_name}_") as td:
		tmp_dir = Path(td)
		log_name = "tmp.log"
		log_path = tmp_dir / log_name

		with log_path.open("w", encoding="utf-8") as f:
			for t in texts:
				f.write(str(t).replace("\n", " ") + "\n")

		if parser_name == "spell":
			from preprocess.Spell import LogParser as SpellParser
			parser = SpellParser(indir=str(tmp_dir), outdir=str(tmp_dir), log_format="<Content>", tau=0.5, rex=[])
		elif parser_name == "logmine":
			from preprocess.LogMine import LogParser as LogMineParser
			parser = LogMineParser(indir=str(tmp_dir), outdir=str(tmp_dir), log_format="<Content>", rex=[])
		elif parser_name == "lke":
			from preprocess.LKE import LogParser as LKEParser
			parser = LKEParser(log_format="<Content>", indir=str(tmp_dir), outdir=str(tmp_dir), rex=[])
		else:
			raise ValueError(f"Unsupported parser for templating: {parser_name}")

		parser.parse(log_name)

		structured_path = tmp_dir / f"{log_name}_structured.csv"
		if not structured_path.exists():
			raise FileNotFoundError(f"Structured output not found: {structured_path}")

		df = pd.read_csv(structured_path)
		if "EventTemplate" not in df.columns:
			raise ValueError(f"EventTemplate column missing in {structured_path}")
		return df["EventTemplate"].astype(str).tolist()


def _preprocess_texts(texts: List[str], dataset_name: str, mode: str) -> List[str]:
	mode = mode.lower()
	if mode in ("raw", "none"):
		return texts

	category, _ = _parse_dataset_meta(dataset_name)

	if mode == "drain":
		from preprocess.drain import DrainParser
		parser = DrainParser(depth=6, st=0.5, registry_mode=(category == "Registry"))
		return [parser.parse(str(t))[0] for t in texts]

	if mode in ("spell", "logmine", "lke"):
		return _template_with_logpai(texts, mode)

	if mode == "ants":
		ants_dir = REPO_ROOT / "ANTS_Share_Preprocessing_Embedding"
		if str(ants_dir) not in sys.path:
			sys.path.insert(0, str(ants_dir))
		from standardizer import standardize

		if category == "File":
			std_type = "file"
		elif category == "Registry":
			std_type = "registry"
		elif category == "Network":
			std_type = "network"
		else:
			raise ValueError(f"Cannot infer ANTS preprocessing type for dataset: {dataset_name}")

		out = []
		for t in texts:
			if std_type in ("file", "registry", "network"):
				out.append(str(standardize(str(t), std_type, mapping_collector=[])))
			else:
				out.append(str(standardize(str(t), std_type)))
		return out

	raise ValueError(f"Unsupported preprocess mode: {mode}")


def evaluate_one_file(
	model,
	model_key: str,
	preprocess_mode: str,
	csv_path: Path,
	batch_size: int,
	normalize: bool,
	max_rows: int | None,
	show_progress: bool,
	save_details: bool,
	output_dir: Path,
) -> EvalResult:
	df = pd.read_csv(csv_path, usecols=["Base", "Near", "Far"]).fillna("")
	if max_rows is not None and max_rows > 0:
		df = df.head(max_rows)

	arr = df[["Base", "Near", "Far"]].astype(str).values
	texts = arr.reshape(-1).tolist()
	texts = _preprocess_texts(texts, csv_path.name, preprocess_mode)

	embs = model.embed(
		texts,
		batch_size=batch_size,
		show_progress=show_progress,
		normalize=normalize,
		dataset_name=csv_path.name,
	)

	n_rows = len(df)
	dim = embs.shape[1]
	trip = embs.reshape(n_rows, 3, dim)

	base = trip[:, 0, :]
	near = trip[:, 1, :]
	far = trip[:, 2, :]

	if normalize:
		# dot product equals cosine for unit-normalized vectors
		sim_near = np.einsum("ij,ij->i", base, near)
		sim_far = np.einsum("ij,ij->i", base, far)
	else:
		sim_near = _safe_cosine(base, near)
		sim_far = _safe_cosine(base, far)

	margin = sim_near - sim_far
	correct_mask = margin > 0
	ties = int(np.sum(margin == 0))
	accuracy = float(np.mean(correct_mask))

	if save_details:
		detail = df.copy()
		detail["preprocess"] = preprocess_mode
		detail["sim_base_near"] = sim_near
		detail["sim_base_far"] = sim_far
		detail["margin"] = margin
		detail["correct"] = correct_mask.astype(int)
		detail_name = f"detail__{model_key}__{preprocess_mode}__{csv_path.stem}.csv"
		detail.to_csv(output_dir / detail_name, index=False)

	return EvalResult(
		model=model_key,
		preprocess=preprocess_mode,
		dataset=csv_path.name,
		n_rows=n_rows,
		accuracy=accuracy,
		ties=ties,
		mean_margin=float(np.mean(margin)),
		median_margin=float(np.median(margin)),
		p10_margin=float(np.percentile(margin, 10)),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Benchmark embedding models on Base/Near/Far triplets")
	parser.add_argument(
		"--dataset-dir",
		type=str,
		default=os.path.join(config.DATA_DIR, "testingdataset"),
		help="Directory containing triplet CSV files with columns Base, Near, Far",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=os.path.join(config.RESULT_DIR, "embedding_model_benchmark"),
		help="Directory to write benchmark outputs",
	)
	parser.add_argument(
		"--models",
		type=str,
		default="all",
		help="Comma-separated model keys, or 'all' to use all registered models",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=64,
		help="Embedding batch size",
	)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=0,
		help="Limit rows per dataset for quick tests (0 = all rows)",
	)
	parser.add_argument(
		"--no-normalize",
		action="store_true",
		help="Disable embedding normalization before similarity",
	)
	parser.add_argument(
		"--show-progress",
		action="store_true",
		help="Show embedding progress bars",
	)
	parser.add_argument(
		"--save-details",
		action="store_true",
		help="Save per-row detail CSV for each model × dataset",
	)
	parser.add_argument(
		"--append-summary",
		action="store_true",
		help="Append/merge into existing summary.csv instead of overwriting it",
	)
	parser.add_argument(
		"--preprocess",
		type=str,
		default="raw",
		help="Preprocessing mode(s): raw, drain, ants, spell, logmine, lke, or comma list, or 'all'",
	)
	args = parser.parse_args()

	dataset_dir = Path(args.dataset_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if not dataset_dir.exists():
		raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

	csv_files = list(_iter_csvs(dataset_dir))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

	model_keys = _resolve_models(args.models)
	preprocess_modes = _resolve_preprocesses(args.preprocess)
	normalize = not args.no_normalize
	max_rows = args.max_rows if args.max_rows > 0 else None

	print("=" * 90)
	print("Embedding Model Benchmark (Base/Near/Far)")
	print("=" * 90)
	print(f"Dataset dir : {dataset_dir}")
	print(f"CSV files   : {len(csv_files)}")
	print(f"Models      : {len(model_keys)} -> {model_keys}")
	print(f"Preprocess  : {preprocess_modes}")
	print(f"Batch size  : {args.batch_size}")
	print(f"Normalize   : {normalize}")
	if max_rows:
		print(f"Max rows    : {max_rows} per CSV")
	print("=" * 90)

	all_rows: List[EvalResult] = []

	for mkey in model_keys:
		print(f"\n[Model] {mkey}")
		try:
			model = get_bert_model(mkey, cache_dir=config.BERT_CACHE_DIR, auto_load=True)
		except Exception as exc:
			print(f"  ! load failed: {exc}")
			continue

		for pre in preprocess_modes:
			print(f"  [Preprocess] {pre}")
			for csv_path in csv_files:
				try:
					r = evaluate_one_file(
						model=model,
						model_key=mkey,
						preprocess_mode=pre,
						csv_path=csv_path,
						batch_size=args.batch_size,
						normalize=normalize,
						max_rows=max_rows,
						show_progress=args.show_progress,
						save_details=args.save_details,
						output_dir=output_dir,
					)
					all_rows.append(r)
					print(
						f"    - {csv_path.name:<43} "
						f"acc={r.accuracy:.4f} | mean_margin={r.mean_margin:.4f} | n={r.n_rows}"
					)
				except Exception as exc:
					print(f"    ! eval failed on {csv_path.name}: {exc}")

		# free model memory before next model
		del model
		gc.collect()

	if not all_rows:
		raise RuntimeError("No successful model evaluations.")

	new_summary = pd.DataFrame([r.__dict__ for r in all_rows])
	new_summary[["category", "strategy"]] = new_summary["dataset"].apply(
		lambda d: pd.Series(_parse_dataset_meta(d))
	)
	summary_path = output_dir / "summary.csv"

	if args.append_summary and summary_path.exists():
		old_summary = pd.read_csv(summary_path)
		if "category" not in old_summary.columns or "strategy" not in old_summary.columns:
			old_summary[["category", "strategy"]] = old_summary["dataset"].apply(
				lambda d: pd.Series(_parse_dataset_meta(d))
			)
		if "preprocess" not in old_summary.columns:
			old_summary["preprocess"] = "raw"
		summary = pd.concat([old_summary, new_summary], ignore_index=True)
		# keep latest result for each model x preprocess x dataset combination
		summary = summary.drop_duplicates(subset=["model", "preprocess", "dataset"], keep="last")
	else:
		summary = new_summary

	summary.to_csv(summary_path, index=False)

	# Weighted global ranking by row count
	ranking = (
		summary
		.groupby(["model", "preprocess"], as_index=False)
		.apply(
			lambda g: pd.Series(
				{
					"total_rows": int(g["n_rows"].sum()),
					"weighted_accuracy": float(np.average(g["accuracy"], weights=g["n_rows"])),
					"weighted_mean_margin": float(np.average(g["mean_margin"], weights=g["n_rows"])),
				}
			),
			include_groups=False
		)
		.reset_index(drop=True)
		.sort_values(["weighted_accuracy", "weighted_mean_margin"], ascending=False)
	)
	ranking_path = output_dir / "model_ranking.csv"
	ranking.to_csv(ranking_path, index=False)

	print("\n" + "=" * 90)
	print("Leaderboard (weighted across all datasets)")
	print("=" * 90)
	for i, row in ranking.head(20).reset_index(drop=True).iterrows():
		label = f"{row['model']}[{row['preprocess']}]"
		print(
			f"{i+1:>2}. {label:<30} "
			f"acc={row['weighted_accuracy']:.4f} "
			f"margin={row['weighted_mean_margin']:.4f} "
			f"rows={int(row['total_rows'])}"
		)

	print("\nOutputs:")
	print(f"- {summary_path}")
	print(f"- {ranking_path}")


if __name__ == "__main__":
	main()
