"""
概念萃取視覺化模組 (Concept Extraction Visualization)

根據 Concept Extraction.md 實作：
1. 模組 A：語義解釋性呈現 - 代表性樣本反查、關鍵字提取、概念定義表
2. 模組 B：結構分離度呈現 - UMAP 降維、3D 散點圖對比

Usage:
    from visualization.conception_extraction_viz import ConceptVisualization
    viz = ConceptVisualization()
    results = viz.run_multi_dataset(n_datasets=5)
"""

import sys
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

warnings.filterwarnings('ignore')

# 調整匯入路徑
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from config import (
    CONCEPT_VECTORS_DIR,
    LOG_VECTORS_DIR,
    INTERMEDIATE_DATA_DIR,
    RESULT_DIR,
)


class ConceptDataLoader:
    """載入概念向量、原始嵌入及日誌文字資料"""
    
    def __init__(
        self,
        concept_vectors_dir: str = CONCEPT_VECTORS_DIR,
        embeddings_dir: str = LOG_VECTORS_DIR,
        intermediate_data_dir: str = INTERMEDIATE_DATA_DIR,
    ):
        self.concept_vectors_dir = Path(PROJECT_ROOT) / concept_vectors_dir
        self.embeddings_dir = Path(PROJECT_ROOT) / embeddings_dir
        self.intermediate_data_dir = Path(PROJECT_ROOT) / intermediate_data_dir
    
    def _load_arrow_as_numpy(self, dir_path: Path, col_name: str) -> Optional[np.ndarray]:
        """從 Arrow/Feather 格式載入指定欄位為 numpy array"""
        arrow_file = dir_path / "data-00000-of-00001.arrow"
        if not arrow_file.exists():
            return None
        
        table = None
        # 嘗試 Feather 格式
        try:
            table = feather.read_table(str(arrow_file))
        except Exception:
            pass
        
        # 嘗試 Arrow IPC Stream 格式
        if table is None:
            try:
                with pa.memory_map(str(arrow_file), 'r') as source:
                    table = ipc.open_stream(source).read_all()
            except Exception:
                pass
        
        # 嘗試 Arrow IPC File 格式
        if table is None:
            try:
                with pa.memory_map(str(arrow_file), 'r') as source:
                    table = ipc.open_file(source).read_all()
            except Exception:
                return None
        
        if table is not None and col_name in table.column_names:
            return np.array(table[col_name].to_pylist())
        return None
    
    def list_available_datasets(self) -> List[str]:
        """列出所有可用的 ConceptVectors 資料集"""
        if not self.concept_vectors_dir.exists():
            return []
        return [
            d.name for d in self.concept_vectors_dir.iterdir() 
            if d.is_dir() and (d / "data-00000-of-00001.arrow").exists()
        ]
    
    def load_concept_vectors(self, dataset_name: str) -> Optional[np.ndarray]:
        """載入 NMF 轉換後的概念向量 (H 矩陣)"""
        return self._load_arrow_as_numpy(self.concept_vectors_dir / dataset_name, "concept_vector")
    
    def load_embeddings(self, dataset_name: str) -> Optional[np.ndarray]:
        """載入原始 BERT 嵌入向量"""
        return self._load_arrow_as_numpy(self.embeddings_dir / dataset_name, "embedding")
    
    def load_raw_logs(self, dataset_name: str) -> Optional[List[str]]:
        """載入原始日誌文字"""
        match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', dataset_name)
        if not match:
            return None
        
        csv_path = self.intermediate_data_dir / f"{match.group(1)}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "ConcatenatedLog" in df.columns:
                    return df["ConcatenatedLog"].tolist()
            except Exception:
                pass
        return None
    
    def load_dataset_complete(self, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """載入完整資料集：概念向量、原始嵌入、日誌文字"""
        return (
            self.load_concept_vectors(dataset_name),
            self.load_embeddings(dataset_name),
            self.load_raw_logs(dataset_name)
        )


class SemanticInterpreter:
    """語義解釋性模組：代表性樣本反查、關鍵字提取、概念定義表"""
    
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
        'this', 'that', 'these', 'those', 'it', 'its', 'success', 'true', 'false',
        'null', 'none', 'nan', 'name', 'result',
    }
    
    def __init__(self, top_n: int = 10, n_keywords: int = 10):
        self.top_n = top_n
        self.n_keywords = n_keywords
    
    def get_representative_samples(
        self, H_matrix: np.ndarray, raw_logs: List[str], top_n: Optional[int] = None
    ) -> Dict[int, List[Tuple[int, str, float]]]:
        """針對每個概念找出權重最高的 Top-N 個日誌樣本"""
        top_n = top_n or self.top_n
        n_concepts = H_matrix.shape[1]
        
        representatives = {}
        for concept_idx in range(n_concepts):
            concept_weights = H_matrix[:, concept_idx]
            sorted_indices = np.argsort(concept_weights)[::-1][:top_n]
            
            samples = [
                (int(idx), raw_logs[idx], float(concept_weights[idx]))
                for idx in sorted_indices if idx < len(raw_logs)
            ]
            representatives[concept_idx] = samples
        
        return representatives
    
    def extract_keywords(
        self, representative_samples: Dict[int, List[Tuple[int, str, float]]], n_keywords: Optional[int] = None
    ) -> Dict[int, List[Tuple[str, float]]]:
        """對每個概念的代表性日誌提取核心關鍵字"""
        n_keywords = n_keywords or self.n_keywords
        concept_keywords = {}
        
        for concept_id, samples in representative_samples.items():
            if not samples:
                concept_keywords[concept_id] = []
                continue
            
            combined_text = " ".join(s[1] for s in samples)
            tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_]{2,}', combined_text.lower())
            filtered = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2 and not t.startswith('0x')]
            
            if not filtered:
                concept_keywords[concept_id] = []
                continue
            
            token_counts = Counter(filtered)
            top_keywords = token_counts.most_common(n_keywords)
            max_count = top_keywords[0][1] if top_keywords else 1
            concept_keywords[concept_id] = [(word, count / max_count) for word, count in top_keywords]
        
        return concept_keywords
    
    def build_concept_definition_table(
        self,
        representative_samples: Dict[int, List[Tuple[int, str, float]]],
        concept_keywords: Dict[int, List[Tuple[str, float]]],
        n_keywords_display: int = 5
    ) -> pd.DataFrame:
        """建立概念定義表"""
        rows = []
        for concept_id in sorted(representative_samples.keys()):
            samples = representative_samples.get(concept_id, [])
            keywords = concept_keywords.get(concept_id, [])
            
            top_kw = ", ".join([kw[0] for kw in keywords[:n_keywords_display]])
            
            if samples:
                rep_log = samples[0][1][:200] + "..." if len(samples[0][1]) > 200 else samples[0][1]
                max_weight = samples[0][2]
            else:
                rep_log, max_weight = "(無樣本)", 0.0
            
            rows.append({
                "Concept_ID": concept_id,
                "Top_Keywords": top_kw or "(無關鍵字)",
                "Representative_Log": rep_log,
                "Max_Weight": round(max_weight, 4)
            })
        
        return pd.DataFrame(rows)


class StructuralVisualizer:
    """結構分離度視覺化模組：UMAP 降維、3D 散點圖"""
    
    def __init__(self, n_components: int = 3, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
    
    def get_dominant_concept_labels(self, H_matrix: np.ndarray) -> np.ndarray:
        """計算 ArgMax(H) 作為每個樣本的標籤"""
        return np.argmax(H_matrix, axis=1)
    
    def reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """UMAP 降維"""
        try:
            import umap
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
                metric='cosine'
            )
            return reducer.fit_transform(X)
        except ImportError:
            from sklearn.decomposition import PCA
            return PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
    
    def create_3d_scatter_plotly(
        self, coords: np.ndarray, labels: np.ndarray, title: str,
        hover_texts: Optional[List[str]] = None, output_path: Optional[Path] = None
    ) -> Any:
        """使用 Plotly 建立互動式 3D 散點圖"""
        try:
            import plotly.express as px
        except ImportError:
            print("⚠️ 缺少 plotly 套件")
            return None
        
        df = pd.DataFrame({
            'UMAP_1': coords[:, 0],
            'UMAP_2': coords[:, 1],
            'UMAP_3': coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(coords)),
            'Concept': labels.astype(str),
            'Text': hover_texts or [f"Sample {i}" for i in range(len(coords))]
        })
        
        fig = px.scatter_3d(df, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='Concept', hover_data=['Text'], title=title, opacity=0.6)
        fig.update_layout(scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'), margin=dict(l=0, r=0, b=0, t=40))
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            print(f"✅ 3D 圖表已儲存至 {output_path}")
        
        return fig
    
    def create_comparison_figure(
        self, original_embeddings: np.ndarray, concept_vectors: np.ndarray, labels: np.ndarray,
        raw_logs: Optional[List[str]] = None, output_dir: Optional[Path] = None
    ) -> Tuple[Any, Any]:
        """並排展示「原始 BERT 空間」與「NMF 概念空間」的 3D 散點圖"""
        print("正在進行 UMAP 降維（原始 BERT 空間）...")
        original_coords = self.reduce_dimensions(original_embeddings)
        
        print("正在進行 UMAP 降維（NMF 概念空間）...")
        nmf_coords = self.reduce_dimensions(concept_vectors)
        
        hover_texts = [log[:100] + "..." if len(log) > 100 else log for log in raw_logs] if raw_logs else None
        
        fig_original = self.create_3d_scatter_plotly(
            original_coords, labels, "原始 BERT 嵌入空間",
            hover_texts, output_dir / "3d_scatter_original_bert.html" if output_dir else None
        )
        fig_nmf = self.create_3d_scatter_plotly(
            nmf_coords, labels, "NMF 概念空間",
            hover_texts, output_dir / "3d_scatter_nmf_concepts.html" if output_dir else None
        )
        
        return fig_original, fig_nmf


class ConceptVisualization:
    """整合語義解釋與結構分離度視覺化的主要管線"""
    
    def __init__(self, output_dir: str = os.path.join(RESULT_DIR, "conception"), top_n_samples: int = 15, n_keywords: int = 10):
        self.output_dir = Path(PROJECT_ROOT) / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = ConceptDataLoader()
        self.semantic = SemanticInterpreter(top_n=top_n_samples, n_keywords=n_keywords)
        self.structural = StructuralVisualizer()
    
    def run_multi_dataset(self, n_datasets: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """對多個資料集執行視覺化流程"""
        available = self.loader.list_available_datasets()
        if not available:
            print("❌ 找不到任何 ConceptVectors 資料集")
            return {}
        
        if n_datasets:
            available = available[:n_datasets]
        
        print(f"📦 找到 {len(available)} 個資料集")
        
        all_results = {}
        for dataset_name in available:
            print(f"\n{'='*60}\n處理資料集: {dataset_name}\n{'='*60}")
            
            H_matrix, embeddings, raw_logs = self.loader.load_dataset_complete(dataset_name)
            results = {"dataset": dataset_name}
            
            if H_matrix is None:
                print(f"⚠️ 無法載入概念向量，跳過")
                continue
            
            print(f"✅ 概念向量載入成功: shape={H_matrix.shape}")
            
            # 模組 A：語義解釋性
            if raw_logs:
                print(f"\n📝 模組 A：語義解釋性分析 (日誌數量: {len(raw_logs)})")
                representatives = self.semantic.get_representative_samples(H_matrix, raw_logs)
                keywords = self.semantic.extract_keywords(representatives)
                definition_table = self.semantic.build_concept_definition_table(representatives, keywords)
                
                table_path = self.output_dir / f"{dataset_name}_concept_definitions.csv"
                definition_table.to_csv(table_path, index=False, encoding='utf-8-sig')
                print(f"   ✅ 概念定義表已儲存至 {table_path}")
                print(f"\n   📊 概念定義表摘要（前 10 個概念）:\n{definition_table.head(10).to_string(index=False)}")
                
                results.update({"representatives": representatives, "keywords": keywords, "definition_table": definition_table})
            else:
                print("⚠️ 無法載入原始日誌，跳過語義分析")
            
            # 模組 B：結構分離度
            if embeddings is not None:
                print(f"\n🔬 模組 B：結構分離度分析 (原始嵌入維度: {embeddings.shape})")
                labels = self.structural.get_dominant_concept_labels(H_matrix)
                print(f"   主導概念分布: {len(np.unique(labels))} 個概念被激活")
                
                fig_orig, fig_nmf = self.structural.create_comparison_figure(
                    embeddings, H_matrix, labels, raw_logs, self.output_dir
                )
                results["3d_figures"] = {"original": fig_orig, "nmf": fig_nmf}
            else:
                print("⚠️ 無法載入原始嵌入，跳過結構分析")
            
            all_results[dataset_name] = results
        
        return all_results


def main():
    """主程式入口"""
    viz = ConceptVisualization()
    
    print("\n" + "="*60)
    print("概念萃取視覺化分析")
    print(f"輸出目錄: {viz.output_dir}")
    print("="*60)
    
    # 顯示原始日誌範例
    available = viz.loader.list_available_datasets()
    if available:
        sample_logs = viz.loader.load_raw_logs(available[0])
        if sample_logs:
            print("\n📋 原始日誌範例（前 3 條）:")
            for i, log in enumerate(sample_logs[:3]):
                print(f"   {i+1}. {log[:150]}..." if len(log) > 150 else f"   {i+1}. {log}")
    
    # 執行分析
    results = viz.run_multi_dataset()
    
    print("\n" + "="*60)
    print(f"✅ 視覺化分析完成，共處理 {len(results)} 個資料集")
    print(f"📁 結果已儲存至: {viz.output_dir}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
