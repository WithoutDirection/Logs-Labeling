"""視覺化模組

整合異常偵測與嵌入比較相關的視覺化工具。
目前所有實作都集中在 :mod:`visualization.anomaly_comparison`。
"""

from visualization.anomaly_comparison import (
    ResultAggregator,
    AggregatedResult,
    plot_trend_analysis,
    plot_distribution_evolution,
    plot_score_histogram,
)

__all__ = [
    "ResultAggregator",
    "AggregatedResult",
    "plot_trend_analysis",
    "plot_distribution_evolution",
    "plot_score_histogram",
]
