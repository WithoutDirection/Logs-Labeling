"""視覺化模組

整合異常偵測、嵌入比較與 TF-IDF 覆蓋分析相關的視覺化工具。

子模組：
    - anomaly_comparison: 異常偵測結果聚合與趨勢分析
    - bert_comparison: BERT 嵌入 UMAP 視覺化比較
    - tfidf_coverage: TF-IDF 詞彙覆蓋率分析 (Venn/Bar Chart)
"""

from visualization.anomaly_comparison import (
    ResultAggregator,
    AggregatedResult,
    plot_trend_analysis,
    plot_distribution_evolution,
    plot_score_histogram,
)

from visualization.tfidf_coverage import (
    run_tfidf_coverage_analysis,
    plot_coverage_bar,
    plot_venn_diagram,
    plot_top_overlapping_terms,
)

# Optional helpers (may be missing in some branches / environments)
try:
    from visualization.stage3_presentation_viz import (
        generate_stage3_presentation_assets,
        list_available_datasets as list_stage3_available_datasets,
    )
except Exception:
    generate_stage3_presentation_assets = None
    list_stage3_available_datasets = None

try:
    from visualization.hmm_mitre_similarity_viz import (
        generate_hmm_mitre_similarity_assets,
        list_available_datasets as list_hmm_mitre_available_datasets,
    )
except Exception:
    generate_hmm_mitre_similarity_assets = None
    list_hmm_mitre_available_datasets = None

__all__ = [
    # anomaly_comparison
    "ResultAggregator",
    "AggregatedResult",
    "plot_trend_analysis",
    "plot_distribution_evolution",
    "plot_score_histogram",
    # tfidf_coverage
    "run_tfidf_coverage_analysis",
    "plot_coverage_bar",
    "plot_venn_diagram",
    "plot_top_overlapping_terms",
    # stage3_presentation_viz (optional)
    "generate_stage3_presentation_assets",
    "list_stage3_available_datasets",
    # hmm_mitre_similarity_viz (optional)
    "generate_hmm_mitre_similarity_assets",
    "list_hmm_mitre_available_datasets",
]
