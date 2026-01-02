"""
Logs Labeling.Pipeline 的 Docstring
整合各個步驟的日誌標記流程
Steps:
1. 預處理 (Preprocessing): Reffer to preprocess/preprocess.py
    配置參數: 
2. 異常檢測 (Anomaly Detection): Reffer to anomaly_detection/log_anomaly_detector.py
    配置參數:
3. 概念提取 (Concept Extraction): Reffer to conception_extraction.py
    配置參數:
4. 序列區塊化 (Sequence Clustering): Reffer to sequence_clustering.py
    配置參數:
"""
import config
import os
import shutil
from utils.path import *
import numpy as np
def main():
    # * 0. 配置資料夾並清除先前實驗結果
    config.DATA_DIR = os.path.join("data")
    config.INPUT_LOGS_DIR = os.path.join(config.DATA_DIR, "input_logs")
    # 清除 data 資料夾中除了 INPUT_LOGS_DIR 以外的所有檔案與資料夾
    if os.path.exists(config.DATA_DIR):
        input_logs_name = os.path.basename(config.INPUT_LOGS_DIR)
        for item in os.listdir(config.DATA_DIR):
            item_path = os.path.join(config.DATA_DIR, item)
            if item != input_logs_name:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"已刪除資料夾: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"已刪除檔案: {item_path}")
    
    # * 1. 配置預處理參數，並執行預處理
    from preprocess.preprocess import LogLoader, LogEmbedder, LogChunker
    # * 1.1 載入日誌
    config.PREPROCESS_ENABLE_PARSER = False
    loader = LogLoader(enable_parser=config.PREPROCESS_ENABLE_PARSER)
    loader.load_logs(num=10) # 只載入前15 Datasets
    
    # * 1.2 日誌嵌入
    config.BERT_MODEL_NAME = "sentence-bert"
    embedder = LogEmbedder(model_name=config.BERT_MODEL_NAME, normalize=False)
    embedder.embed_logs()
    
    # # * 1.3 日誌區塊化
    # chnuker = LogChunker(
    #     window_size=config.SEQUENCE_WINDOW_SIZE,
    #     stride=config.SEQUENCE_STRIDE,
    #     bilstm_hidden_size=config.BILSTM_HIDDEN_SIZE,
    #     bilstm_num_layers=config.BILSTM_NUM_LAYERS,
    #     bilstm_dropout=config.BILSTM_DROPOUT,
    #     fusion_enable=config.FUSION_ENABLE,
    #     fusion_output_dim=config.FUSION_OUTPUT_DIM,
    # )
    # chnuker.chunk_logs()

    # # * Result of Stage 1
    from visualization.bert_comparison import BertEmbeddingComparator
    comparator = BertEmbeddingComparator(
        model_keys=['securebert', 'sentence-bert', 'bert-base-nli'],
        max_samples=1000,
    )
    comparator.run(n=10)
    
    # * 2. 執行異常檢測（批次模式：一次載入所有數據、合併訓練、整體視覺化）
    from anomaly_dection.log_detector import run_detection_pipeline, generate_detection_summary
    results = run_detection_pipeline(
        input_dir=config.LOG_VECTORS_DIR,
        output_dir=config.DETECTION_RESULTS_DIR,
        models=getattr(config, "DETECTION_MODELS", None),
        verbose=True
    )
    if results:
        # 生成整體視覺化與計算相關結果
        generate_detection_summary(
            results, 
            output_dir=config.DETECTION_VIZ_DIR,
            generate_visualizations=True,
            enable_advanced_plots=True
        )
        
    # TODO: 跟據檢測結果產生log vector在後續標示階段的weight
    
    # * 3. 執行概念提取
    from conception_extraction import ConceptExtractor, train_concept_extractor, transform_all_datasets
    extractor = ConceptExtractor()
    extractor = train_concept_extractor()
    transform_all_datasets()
    
    # * 4. 執行序列區塊化
    from sequence_clustering import SequenceClustering, load_concept_vectors
    print("=" * 60)
    print("序列分群 - Per-Dataset HMM 策略")
    print("=" * 60)
    
    vectors = load_concept_vectors()
    
    from visualization.conception_extraction_viz import ConceptVisualization
    viz = ConceptVisualization(output_dir=os.path.join(config.RESULT_DIR, "conception_sequence_clustering"))
    viz.run_multi_dataset(n_datasets=5)
    
    # ===== 批次處理所有資料集 =====
    clusterer = SequenceClustering()
    results = clusterer.batch_process_all(vectors)
    
    if results:
        avg_clusters = np.mean([len(np.unique(labels)) for labels in results.values()])
        print(f"平均群集數: {avg_clusters:.2f}")
    
    print("\n[完成] 序列分群已完成。")
    
    

if __name__ == "__main__":
    main()