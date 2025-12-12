"""視覺化模組

提供多規模異常偵測結果的視覺化功能：
- aggregator: 結果聚合與標準化
- trend_analysis: 效能趨勢分析
- distribution_plot: 分佈演變圖
"""

from visualization.aggregator import ResultAggregator, AggregatedResult
from visualization.trend_analysis import plot_trend_analysis
from visualization.distribution_plot import plot_distribution_evolution, plot_score_histogram

__all__ = [
    "ResultAggregator",
    "AggregatedResult",
    "plot_trend_analysis",
    "plot_distribution_evolution",
    "plot_score_histogram"
]
