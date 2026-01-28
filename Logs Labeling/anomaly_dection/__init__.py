"""異常偵測模組

提供多種非監督式異常偵測演算法：
- Isolation Forest：幾何隔離法
- COPOD：統計機率法
- AutoEncoder：重構誤差法
- PCA + GMM：機率密度估計法

主要 API:
    run_detection(verbose=True) - 執行完整異常偵測流程
"""
import config
from anomaly_dection.isolation_forest import IsolationForestDetector, IsolationForestConfig
from anomaly_dection.copod import COPODDetector, COPODConfig
from anomaly_dection.autoencoder import AutoEncoderDetector, AutoEncoderConfig
from anomaly_dection.pca_gmm import PCAGMMDetector, PCAGMMConfig
from anomaly_dection.log_detector import (
    LogDetector, 
    LogDetectorConfig, 
    ScalerType, 
    ThresholdMethod,
    run_detection_pipeline
)

__all__ = [
    # Isolation Forest
    "IsolationForestDetector",
    "IsolationForestConfig",
    # COPOD
    "COPODDetector",
    "COPODConfig",
    # AutoEncoder
    "AutoEncoderDetector",
    "AutoEncoderConfig",
    # PCA + GMM
    "PCAGMMDetector",
    "PCAGMMConfig",
    # Log Detector
    "LogDetector",
    "LogDetectorConfig",
    "ScalerType",
    "ThresholdMethod",
    # Pipeline API
    "run_detection_pipeline",
    "run_detection"
]


def run_detection(
    input_dir: str = None,
    output_dir: str = None,
    viz_dir: str = None,
    models: list = None,
    generate_viz: bool = True,
    verbose: bool = True
) -> dict:
    """
    執行完整的異常偵測流程
    
    整合偵測與視覺化，提供一站式 API。
    
    Args:
        input_dir: Log Vector 輸入目錄（預設 config.LOG_VECTORS_DIR）
        output_dir: 偵測結果輸出目錄（預設 config.DETECTION_RESULTS_DIR）
        viz_dir: 視覺化輸出目錄（預設 config.DETECTION_VIZ_DIR）
        models: 要使用的模型列表（預設使用 config.DETECTION_MODELS）
        generate_viz: 是否生成視覺化報告
        verbose: 是否顯示詳細資訊
        
    Returns:
        包含處理結果的字典：
        - results: 各資料集的偵測結果
        - n_datasets: 處理的資料集數量
        - models: 使用的模型列表
        
    Example:
        >>> from anomaly_dection import run_detection
        >>> result = run_detection()
        >>> print(f"處理了 {result['n_datasets']} 個資料集")
    """
    from visualization.anomaly_comparison import generate_detection_summary
    
    input_dir = input_dir or config.LOG_VECTORS_DIR
    output_dir = output_dir or config.DETECTION_RESULTS_DIR
    viz_dir = viz_dir or getattr(config, 'DETECTION_VIZ_DIR', None)
    models = models or config.DETECTION_MODELS
    
    # 執行偵測
    results = run_detection_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        models=models,
        verbose=verbose
    )
    
    # 生成視覺化
    if results and generate_viz and viz_dir:
        if verbose:
            print("\n生成視覺化報告...")
        generate_detection_summary(
            results,
            output_dir=viz_dir,
            generate_visualizations=True,
            enable_advanced_plots=True
        )
    
    return {
        "results": results,
        "n_datasets": len(results) if results else 0,
        "models": models
    }
