"""
BERT 嵌入模型比較視覺化工具

功能說明：
    比較不同 BERT 模型的嵌入效果，使用 UMAP 降維至 3D 視覺化並計算分群品質指標。

參數說明：
    -n, --num-datasets      要比較的資料集數量 (預設: 5)
    --models                要比較的模型列表，以空格分隔
                            可選: codebert, securebert, secbert, sentence-bert, 
                                  sentence-bert-large, bert-base, cti-bert
    --max-samples           每個資料集最大取樣數 (預設: 200)
    --list-models           列出所有可用模型

使用範例：
    # 比較指定模型
    python bert_comparison.py --models codebert securebert sentence-bert -n 5
    
    # 使用更多樣本
    python bert_comparison.py --models sentence-bert sentence-bert-large -n 10 --max-samples 500
    
    # 列出可用模型
    python bert_comparison.py --list-models

輸出：
    result/bert_comparison/
    ├── umap_comparison_3d.html          # 所有模型互動式 3D UMAP 圖
    ├── {model_name}_umap_3d.html       # 各模型獨立互動式 3D UMAP 圖
    ├── dispersion_comparison.png        # 分散程度比較圖
    └── model_comparison_statistics.csv  # 完整統計數據
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False

# 路徑設定
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(WORKSPACE_ROOT))

from utils.path import join_path, get_stem, get_filtered_files, ensure_dir
import config

# 預設配置
INTERMEDIATE_DATA_DIR = config.INTERMEDIATE_DATA_DIR
OUTPUT_DIR = join_path("result", "bert_comparison")

# 預設比較模型
DEFAULT_MODELS = ['sentence-bert', 'sentence-bert-large', 'secbert']


class BertEmbeddingComparator:
    """BERT 嵌入比較器"""
    
    def __init__(
        self,
        model_keys: List[str] = None,
        output_dir: str = OUTPUT_DIR,
        max_samples: int = 200,
    ):
        """
        初始化比較器
        
        Args:
            model_keys: 要比較的模型列表
            output_dir: 輸出目錄
            max_samples: 每資料集最大樣本數
        """
        self.model_keys = model_keys or DEFAULT_MODELS
        self.output_dir = output_dir
        self.max_samples = max_samples
        
        self.models = {}
        self.embeddings = {}
        self.texts = {}
        self.stats = None
    
    def _load_models(self):
        """載入 BERT 模型"""
        from models.bert import get_bert_model
        
        print("=" * 60)
        print("載入 BERT 模型")
        print("=" * 60)
        
        for key in self.model_keys:
            print(f"\n載入: {key}")
            try:
                self.models[key] = get_bert_model(key, auto_load=True)
                print(f"  ✓ 成功，維度: {self.models[key].get_embedding_dim()}")
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
    
    def _load_datasets(self, n: int) -> Dict[str, List[str]]:
        """載入資料集文本"""
        print(f"\n載入 {n} 個資料集...")
        
        files = get_filtered_files(INTERMEDIATE_DATA_DIR, ".csv", num=n)
        
        for file in tqdm(files, desc="讀取資料集"):
            path = join_path(INTERMEDIATE_DATA_DIR, file)
            name = get_stem(file)
            
            try:
                df = pd.read_csv(path, encoding='utf-8')
                
                # 依優先順序選擇文本欄位
                for col in ['ConcatenatedLog', 'OriginalLog', 'Template']:
                    if col in df.columns:
                        texts = df[col].fillna("").astype(str).tolist()
                        break
                else:
                    continue
                
                # 限制樣本數
                if len(texts) > self.max_samples:
                    idx = np.linspace(0, len(texts)-1, self.max_samples, dtype=int)
                    texts = [texts[i] for i in idx]
                
                self.texts[name] = texts
            except Exception as e:
                print(f"  ✗ {file}: {e}")
        
        print(f"\n成功載入 {len(self.texts)} 個資料集")
        return self.texts
    
    def _compute_embeddings(self):
        """計算嵌入向量"""
        print("\n" + "=" * 60)
        print("計算嵌入向量")
        print("=" * 60)
        
        self.embeddings = {key: {} for key in self.models}
        
        for model_key, model in self.models.items():
            print(f"\n{model_key}:")
            for name, texts in tqdm(self.texts.items(), desc=f"  {model_key}"):
                self.embeddings[model_key][name] = model.embed(texts, normalize=True)
    
    def _compute_statistics(self) -> pd.DataFrame:
        """計算分散程度統計"""
        print("\n計算統計指標...")
        
        records = []
        for model_key in self.models:
            for name in self.texts:
                emb = self.embeddings[model_key][name]
                
                # 計算指標
                pairwise = pdist(emb, metric='cosine')
                centroid = np.mean(emb, axis=0)
                
                records.append({
                    'Model': model_key,
                    'Dataset': name[:12] + '...' if len(name) > 12 else name,
                    'Dataset_Full': name,
                    'N_Samples': len(emb),
                    'Embedding_Dim': emb.shape[1],
                    'Mean_Pairwise_Dist': float(np.mean(pairwise)),
                    'Std_Pairwise_Dist': float(np.std(pairwise)),
                    'Centroid_Dist': float(np.mean(np.linalg.norm(emb - centroid, axis=1))),
                    'Variance': float(np.var(emb)),
                })
        
        self.stats = pd.DataFrame(records)
        return self.stats
    
    def _compute_silhouette(self) -> pd.DataFrame:
        """計算 Silhouette Score"""
        print("計算 Silhouette Score...")
        
        results = []
        for model_key in self.models:
            all_emb, labels = [], []
            
            for idx, name in enumerate(self.texts):
                emb = self.embeddings[model_key][name]
                all_emb.append(emb)
                labels.extend([idx] * len(emb))
            
            all_emb = np.vstack(all_emb)
            
            try:
                score = silhouette_score(all_emb, labels, metric='cosine')
            except:
                score = 0.0
            
            results.append({
                'Model': model_key,
                'Silhouette_Score': score,
                'Interpretation': 'Excellent' if score > 0.5 else ('Good' if score > 0.25 else 'Fair')
            })
        
        return pd.DataFrame(results)
    
    def _run_umap(self, n_neighbors: int = 15, min_dist: float = 0.1) -> Dict[str, np.ndarray]:
        """執行 UMAP 3D 降維"""
        print("\n" + "=" * 60)
        print("執行 UMAP 3D 降維")
        print("=" * 60)
        
        results = {}
        
        for model_key in self.models:
            print(f"\n{model_key}:")
            
            combined = np.vstack([self.embeddings[model_key][n] for n in self.texts])
            
            # 執行 UMAP 降維至 3D
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(n_neighbors, len(combined)-1),
                min_dist=min_dist,
                metric='cosine',
                random_state=42
            )
            
            results[model_key] = reducer.fit_transform(combined)
            print(f"  ✓ 完成: {results[model_key].shape}")
        
        return results
    
    def _generate_unique_colors(self, n: int) -> List[str]:
        """生成 n 個不重複的顏色"""
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
        return colors
    
    def _plot_umap_comparison(self, umap_results: Dict[str, np.ndarray], save_path: str):
        """繪製互動式 3D UMAP 對比圖"""
        n_models = len(umap_results)
        names = list(self.texts.keys())
        colors = self._generate_unique_colors(len(names))
        
        # 創建子圖
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=[f'{model}<br>(dim={self.models[model].get_embedding_dim()})' 
                           for model in umap_results.keys()],
            specs=[[{'type': 'scatter3d'} for _ in range(n_models)]],
            horizontal_spacing=0.05
        )
        
        for col_idx, (model, emb) in enumerate(umap_results.items(), start=1):
            start = 0
            for idx, name in enumerate(names):
                end = start + len(self.texts[name])
                texts = self.texts[name]
                
                # 添加 3D 散點圖
                fig.add_trace(
                    go.Scatter3d(
                        x=emb[start:end, 0],
                        y=emb[start:end, 1],
                        z=emb[start:end, 2],
                        mode='markers',
                        name=name[:15] + '...' if len(name) > 15 else name,
                        text=[f'Dataset: {name}<br>Log: {text[:100]}...' if len(text) > 100 else f'Dataset: {name}<br>Log: {text}' 
                              for text in texts],
                        hovertemplate='<b>%{text}</b><extra></extra>',
                        marker=dict(
                            size=5,
                            color=colors[idx],
                            opacity=0.7,
                            line=dict(color='white', width=0.5)
                        ),
                        showlegend=(col_idx == 1)  # 只在第一個子圖顯示圖例
                    ),
                    row=1, col=col_idx
                )
                start = end
        
        # 更新布局
        fig.update_layout(
            title_text='BERT Embedding UMAP 3D Comparison',
            title_font_size=16,
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                font=dict(size=10)
            )
        )
        
        # 更新 3D 場景設置
        for i in range(1, n_models + 1):
            fig.update_scenes(
                dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                row=1, col=i
            )
        
        fig.write_html(save_path)
        print(f"\n互動式圖表儲存: {save_path}")
    
    def _plot_umap_per_model(self, umap_results: Dict[str, np.ndarray]):
        """為每個模型繪製獨立互動式 3D UMAP 圖"""
        names = list(self.texts.keys())
        colors = self._generate_unique_colors(len(names))
        
        print("\n生成各模型 3D UMAP 圖...")
        
        for model, emb in umap_results.items():
            fig = go.Figure()
            
            start = 0
            for idx, name in enumerate(names):
                end = start + len(self.texts[name])
                texts = self.texts[name]
                
                # 添加 3D 散點圖，每個資料點有唯一顏色
                for i in range(start, end):
                    point_color = colors[idx]
                    fig.add_trace(
                        go.Scatter3d(
                            x=[emb[i, 0]],
                            y=[emb[i, 1]],
                            z=[emb[i, 2]],
                            mode='markers',
                            name=f'{name}_{i-start}',
                            text=f'Dataset: {name}<br>Index: {i-start}<br>Log: {texts[i-start][:200]}...' 
                                 if len(texts[i-start]) > 200 
                                 else f'Dataset: {name}<br>Index: {i-start}<br>Log: {texts[i-start]}',
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            marker=dict(
                                size=6,
                                color=point_color,
                                opacity=0.8,
                                line=dict(color='white', width=0.5)
                            ),
                            showlegend=False,
                            legendgroup=name,
                        )
                    )
                
                # 添加圖例代表
                fig.add_trace(
                    go.Scatter3d(
                        x=[None],
                        y=[None],
                        z=[None],
                        mode='markers',
                        name=name[:20] + '...' if len(name) > 20 else name,
                        marker=dict(
                            size=10,
                            color=colors[idx],
                            opacity=0.8
                        ),
                        legendgroup=name,
                        showlegend=True
                    )
                )
                
                start = end
            
            dim = self.models[model].get_embedding_dim()
            fig.update_layout(
                title=f'{model.upper()} UMAP 3D Visualization (dim={dim})',
                scene=dict(
                    xaxis_title='UMAP Dimension 1',
                    yaxis_title='UMAP Dimension 2',
                    zaxis_title='UMAP Dimension 3',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=1200,
                height=900,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=10)
                ),
                hovermode='closest'
            )
            
            path = join_path(self.output_dir, f"{model}_umap_3d.html")
            fig.write_html(path)
            print(f"  ✓ {model}: {path}")
    
    def _plot_dispersion(self, save_path: str):
        """繪製分散程度比較圖"""
        if self.stats is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['Mean_Pairwise_Dist', 'Std_Pairwise_Dist', 'Centroid_Dist', 'Variance']
        titles = ['Mean Pairwise Distance', 'Std of Pairwise Distance', 
                  'Mean Centroid Distance', 'Embedding Variance']
        
        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            pivot = self.stats.pivot(index='Dataset', columns='Model', values=metric)
            x = np.arange(len(pivot))
            width = 0.8 / len(self.models)
            
            for i, model in enumerate(self.models):
                if model in pivot.columns:
                    offset = (i - len(self.models)/2 + 0.5) * width
                    ax.bar(x + offset, pivot[model], width, label=model, alpha=0.8)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Embedding Dispersion Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖表儲存: {save_path}")
        plt.close()
    
    def run(self, n: int = 5) -> Dict:
        """
        執行完整比較流程
        
        Args:
            n: 資料集數量
            
        Returns:
            包含統計結果的字典
        """
        print("=" * 70)
        print("BERT 嵌入模型比較")
        print(f"模型: {self.model_keys}")
        print(f"資料集數量: {n}")
        print("=" * 70)
        
        # 1. 載入模型
        self._load_models()
        if not self.models:
            print("無可用模型")
            return None
        
        # 2. 載入資料集
        self._load_datasets(n)
        if not self.texts:
            print("無可用資料集")
            return None
        
        # 3. 計算嵌入
        self._compute_embeddings()
        
        # 4. 計算統計
        stats = self._compute_statistics()
        silhouette = self._compute_silhouette()
        
        # 5. UMAP 3D 降維
        umap_results = self._run_umap()
        
        # 6. 輸出結果
        print("\n" + "=" * 70)
        print("結果摘要")
        print("=" * 70)
        
        summary = stats.groupby('Model').agg({
            'N_Samples': 'sum',
            'Embedding_Dim': 'first',
            'Mean_Pairwise_Dist': 'mean',
            'Centroid_Dist': 'mean',
            'Variance': 'mean',
        }).round(4)
        
        print("\n各模型平均分散程度:")
        print(summary.to_string())
        print("\n\n分群品質 (Silhouette Score):")
        print(silhouette.to_string(index=False))
        
        # 7. 繪製圖表
        print("\n繪製圖表...")
        ensure_dir(self.output_dir)
        
        self._plot_umap_comparison(umap_results, join_path(self.output_dir, "umap_comparison_3d.html"))
        self._plot_umap_per_model(umap_results)
        self._plot_dispersion(join_path(self.output_dir, "dispersion_comparison.png"))
        
        # 8. 儲存統計
        stats_path = join_path(self.output_dir, "model_comparison_statistics.csv")
        stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"\n統計儲存: {stats_path}")
        
        print("\n" + "=" * 70)
        print("完成！")
        print(f"輸出目錄: {self.output_dir}")
        print("=" * 70)
        
        return {'statistics': stats, 'silhouette': silhouette, 'umap': umap_results}


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='BERT 嵌入模型比較工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例：
  %(prog)s --models codebert securebert sentence-bert -n 5
  %(prog)s --models sentence-bert sentence-bert-large -n 10 --max-samples 500
  %(prog)s --list-models
        '''
    )
    parser.add_argument('-n', '--num-datasets', type=int, default=5,
                        help='資料集數量 (預設: 5)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='要比較的模型列表')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='每資料集最大樣本數 (預設: 200)')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用模型')
    
    args = parser.parse_args()
    
    if args.list_models:
        try:
            from models.bert import list_available_models
            print("\n可用模型:")
            for key, desc in list_available_models().items():
                print(f"  {key:25s} - {desc}")
        except ImportError as e:
            print(f"載入失敗: {e}")
        return
    
    comparator = BertEmbeddingComparator(
        model_keys=args.models,
        max_samples=args.max_samples,
    )
    comparator.run(n=args.num_datasets)


if __name__ == "__main__":
    main()
