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
import ast
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


PREPROCESS_MODES = (
	"raw",
	"drain",
	"drain-params",
	"ants",
	"spell",
	"spell-params",
	"logmine",
	"lke",
)
DEFAULT_ALL_PREPROCESS_MODES = tuple(m for m in PREPROCESS_MODES if m != "lke")
PREPROCESS_ALIASES = {"none": "raw"}
SUMMARY_COLUMNS = [
	"model",
	"preprocess",
	"dataset",
	"n_rows",
	"accuracy",
	"ties",
	"mean_margin",
	"median_margin",
	"p10_margin",
	"category",
	"strategy",
]
CATEGORY_TO_ANTS_TYPE = {
	"File": "file",
	"Registry": "registry",
	"Network": "network",
}


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


def _resolve_preprocesses(pre_arg: str, include_lke_in_all: bool = False) -> List[str]:
	if pre_arg.strip().lower() == "all":
		if include_lke_in_all:
			return list(PREPROCESS_MODES)
		return list(DEFAULT_ALL_PREPROCESS_MODES)

	requested: List[str] = []
	for mode in (p.strip().lower() for p in pre_arg.split(",")):
		if not mode:
			continue
		resolved = PREPROCESS_ALIASES.get(mode, mode)
		if resolved not in PREPROCESS_MODES:
			valid = ", ".join([*PREPROCESS_MODES, *PREPROCESS_ALIASES.keys(), "all"])
			raise ValueError(f"Unknown preprocess mode: {mode}. Valid: {valid}")
		requested.append(resolved)

	if not requested:
		raise ValueError("No preprocess mode provided.")
	return requested


def _iter_csvs(dataset_dir: Path) -> Iterable[Path]:
	return sorted(p for p in dataset_dir.glob("*.csv") if p.is_file())


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""Row-wise cosine similarity."""
	a_norm = np.linalg.norm(a, axis=1)
	b_norm = np.linalg.norm(b, axis=1)
	denom = np.clip(a_norm * b_norm, 1e-12, None)
	return np.sum(a * b, axis=1) / denom


def _format_template_with_params(template: str, params: List[str]) -> str:
	params_text = " | ".join(str(p) for p in params if str(p).strip())
	if not params_text:
		return str(template)
	return f"{template} | params: {params_text}"


def _parse_parameter_list_cell(v) -> List[str]:
	if v is None or (isinstance(v, float) and pd.isna(v)):
		return []
	if isinstance(v, list):
		return [str(x) for x in v]
	s = str(v).strip()
	if not s:
		return []
	try:
		parsed = ast.literal_eval(s)
		if isinstance(parsed, (list, tuple)):
			return [str(x) for x in parsed]
		return [str(parsed)]
	except Exception:
		return [s]


def _template_with_logpai(texts: List[str], parser_name: str, include_params: bool = False) -> List[str]:
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

		if include_params:
			if parser_name != "spell":
				raise ValueError(f"include_params is only supported for spell, got: {parser_name}")
			if "ParameterList" not in df.columns:
				raise ValueError(f"ParameterList column missing in {structured_path}")
			out = []
			for _, row in df.iterrows():
				tmpl = str(row["EventTemplate"])
				params = _parse_parameter_list_cell(row["ParameterList"])
				out.append(_format_template_with_params(tmpl, params))
			return out

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

	if mode == "drain-params":
		from preprocess.drain import DrainParser
		parser = DrainParser(depth=6, st=0.5, registry_mode=(category == "Registry"))
		out = []
		for t in texts:
			tmpl, params = parser.parse(str(t))
			out.append(_format_template_with_params(tmpl, params))
		return out

	if mode in ("spell", "logmine", "lke"):
		return _template_with_logpai(texts, mode)

	if mode == "spell-params":
		return _template_with_logpai(texts, "spell", include_params=True)

	if mode == "ants":
		ants_dir = REPO_ROOT / "ANTS_Share_Preprocessing_Embedding"
		if str(ants_dir) not in sys.path:
			sys.path.insert(0, str(ants_dir))
		from standardizer import standardize
		std_type = CATEGORY_TO_ANTS_TYPE.get(category)
		if std_type is None:
			raise ValueError(f"Cannot infer ANTS preprocessing type for dataset: {dataset_name}")
		return [str(standardize(str(t), std_type, mapping_collector=[])) for t in texts]

	raise ValueError(f"Unsupported preprocess mode: {mode}")


def _load_or_init_summary(summary_path: Path, append_summary: bool) -> pd.DataFrame:
	if not append_summary or not summary_path.exists():
		return pd.DataFrame(columns=SUMMARY_COLUMNS)

	summary = pd.read_csv(summary_path)
	required_columns = {
		"model",
		"preprocess",
		"dataset",
		"n_rows",
		"accuracy",
		"ties",
		"mean_margin",
		"median_margin",
		"p10_margin",
	}
	missing = sorted(required_columns - set(summary.columns))
	if missing:
		raise ValueError(f"summary.csv missing required columns: {missing}")

	if "category" not in summary.columns or "strategy" not in summary.columns:
		parsed = summary["dataset"].astype(str).apply(_parse_dataset_meta)
		summary["category"] = parsed.map(lambda x: x[0])
		summary["strategy"] = parsed.map(lambda x: x[1])

	return summary.reindex(columns=SUMMARY_COLUMNS)


def _upsert_summary_row(summary: pd.DataFrame, result: EvalResult) -> pd.DataFrame:
	category, strategy = _parse_dataset_meta(result.dataset)
	new_row = pd.DataFrame(
		[
			{
				"model": result.model,
				"preprocess": result.preprocess,
				"dataset": result.dataset,
				"n_rows": result.n_rows,
				"accuracy": result.accuracy,
				"ties": result.ties,
				"mean_margin": result.mean_margin,
				"median_margin": result.median_margin,
				"p10_margin": result.p10_margin,
				"category": category,
				"strategy": strategy,
			}
		]
	)
	if summary.empty:
		combined = new_row
	else:
		combined = pd.concat([summary, new_row], ignore_index=True)
	return combined.drop_duplicates(subset=["model", "preprocess", "dataset"], keep="last")


def _build_model_ranking(summary: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for (model, preprocess), group in summary.groupby(["model", "preprocess"], as_index=False):
		weights = group["n_rows"].to_numpy(dtype=float)
		rows.append(
			{
				"model": model,
				"preprocess": preprocess,
				"total_rows": int(group["n_rows"].sum()),
				"weighted_accuracy": float(np.average(group["accuracy"], weights=weights)),
				"weighted_mean_margin": float(np.average(group["mean_margin"], weights=weights)),
			}
		)

	ranking = pd.DataFrame(rows)
	if ranking.empty:
		return ranking

	return ranking.sort_values(["weighted_accuracy", "weighted_mean_margin"], ascending=False).reset_index(drop=True)


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
	n_rows = len(df)
	if n_rows == 0:
		raise ValueError(f"Dataset has no rows: {csv_path}")

	embs = model.embed(
		texts,
		batch_size=batch_size,
		show_progress=show_progress,
		normalize=normalize,
		dataset_name=csv_path.name,
	)

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
		help="Preprocessing mode(s): raw, drain, drain-params, ants, spell, spell-params, logmine, lke, comma list, or 'all' (all excludes lke unless --include-lke-in-all)",
	)
	parser.add_argument(
		"--include-lke-in-all",
		action="store_true",
		help="When --preprocess all is used, include lke as well (disabled by default because lke is very slow).",
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
	preprocess_modes = _resolve_preprocesses(args.preprocess, include_lke_in_all=args.include_lke_in_all)
	normalize = not args.no_normalize
	max_rows = args.max_rows if args.max_rows > 0 else None
	summary_path = output_dir / "summary.csv"
	summary = _load_or_init_summary(summary_path, args.append_summary)
	if args.preprocess.strip().lower() == "all" and not args.include_lke_in_all:
		print("[Info] Skipping lke in '--preprocess all' for speed. Use --include-lke-in-all to include it.")

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

	for mkey in model_keys:
		print(f"\n[Model] {mkey}")
		model = get_bert_model(mkey, cache_dir=config.BERT_CACHE_DIR, auto_load=True)

		for pre in preprocess_modes:
			print(f"  [Preprocess] {pre}")
			for csv_path in csv_files:
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
				print(
					f"    - {csv_path.name:<43} "
					f"acc={r.accuracy:.4f} | mean_margin={r.mean_margin:.4f} | n={r.n_rows}"
				)
				summary = _upsert_summary_row(summary, r)
				summary.to_csv(summary_path, index=False)

		# free model memory before next model
		del model
		gc.collect()

	if summary.empty:
		raise RuntimeError("No evaluations were written to summary.csv.")

	# Weighted global ranking by row count
	ranking = _build_model_ranking(summary)
	if ranking.empty:
		raise RuntimeError("Unable to compute model ranking from summary.csv.")
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
