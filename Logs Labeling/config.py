import os

# 資料夾路徑
DATA_DIR = os.path.join("data")
INPUT_LOGS_DIR = os.path.join(DATA_DIR, "input_logs")
REFERENCE_RESOURCES_DIR = os.path.join(DATA_DIR, "reference_resources")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "Intermediate_data")
PROCESSED_LOGS_DIR = os.path.join(DATA_DIR, "processed_logs")
BERT_MODEL_DIR = os.path.join("models", "bert_model")
UNSUPERVISED_MODEL_DIR = os.path.join("models", "Unsupervised")
HMM_MODEL_DIR = os.path.join("models", "HMM")

# 預處理相關設定
ENABLE_PARSER = True  # Whether to parse raw events (True) or keep original (False)
PARSER_LIST = ["drain", "spell", "lenma"]  # Available parser modules (lowercase)
DEFAULT_PARSER = "drain"

# Zipf's Law Preprocessing 
ZIPF_PERCENTILE = 0.05  # Filter top 5% high-frequency words
USE_LOG_HIGH_FREQ = True  # Include common log words (Authority, System, etc.)

# 參數設定
# BERT Model Configuration
# Available options: 'sentence-bert', 'sentence-bert-multilingual', 'sentence-bert-large',
#                   'bert-base-nli', 'bert-base', 'distilbert', 'roberta'
# Or use custom model name from HuggingFace
BERT_MODEL_NAME = "sentence-bert"
BERT_CACHE_DIR = os.path.join(BERT_MODEL_DIR, "cache")  # Cache directory for downloaded models
BERT_AUTO_LOAD = True  # Whether to automatically load model on initialization



CHUNK_MODEL_PATH = os.path.join(BERT_MODEL_DIR, "chunk_model.pt")

# LogChunker 設定
SEQUENCE_WINDOW_SIZE = 5  # 視窗大小
SEQUENCE_STRIDE = 3  # 滑動步長 (重疊量 = window_size - stride)
BILSTM_HIDDEN_SIZE = 128  # BiLSTM 隱藏層維度
BILSTM_NUM_LAYERS = 2  # BiLSTM 層數
BILSTM_DROPOUT = 0.3  # Dropout 比例
FUSION_ENABLE = False  # 是否啟用融合層
FUSION_OUTPUT_DIM = 256  # 融合層輸出維度
LOG_VECTORS_DIR = os.path.join(DATA_DIR, "LogVectors")  # Log Vector 儲存目錄

# ==================== 異常偵測設定 ====================

# Isolation Forest 設定
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.05  # 預期異常比例，可設為 "auto"
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42

# COPOD 設定
COPOD_CONTAMINATION = 0.05

# AutoEncoder 設定
AE_LATENT_DIM = 32
AE_HIDDEN_DIMS = [128, 64]
AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3

# PCA + GMM 設定
PCA_EXPLAINED_VAR = 0.95
GMM_N_COMPONENTS_RANGE = (2, 10)
GMM_COVARIANCE_TYPE = "full"
GMM_USE_BIC = True

# Log Detector 整合設定
DETECTION_MODELS = ["isolation_forest", "copod", "autoencoder", "pca_gmm"]
SCORE_SCALER = "minmax"  # 可選: "minmax", "rank", "zscore"
THRESHOLDING_METHOD = "percentile"  # 可選: "percentile", "std", "top_n"
THRESHOLDING_PARAMS = {"percentile": 98}  # 根據方法調整
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.25,
    "copod": 0.25,
    "autoencoder": 0.25,
    "pca_gmm": 0.25
}
DETECTION_RESULTS_DIR = os.path.join(DATA_DIR, "Detection_Results")

# 舊版相容設定
ISOLATION_FOREST_PARAMS = {
	"n_estimators": IF_N_ESTIMATORS,
	"contamination": IF_CONTAMINATION,
	"random_state": IF_RANDOM_STATE
}
NMF_COMPONENTS = 50 ## 設定的提取概念數量
NMF_MAX_ITER = 1000
NMF_TOL = 1e-4
NMF_INIT = "nndsvd"
CONCEPT_SAMPLE_RATIO = 1.0 # 用於 NMF 訓練的樣本比例
NMF_MODEL_PATH = os.path.join("models", "nmf_concept_model.pkl")
CONCEPT_VECTORS_DIR = os.path.join(DATA_DIR, "ConceptVectors")
EXTERNAL_KNOWLEDGE_DIR = os.path.join(DATA_DIR, "ExternalKnowledge")
HMM_STATES = 3

# ==================== 序列分群設定 (Sequence Clustering) ====================
# 採用 Per-Dataset HMM 策略：為每個資料集獨立訓練 HMM 模型

# HMM 模型架構參數
HMM_K_MIN = 1  # 隱藏狀態數量下界
HMM_K_MAX = 10  # 隱藏狀態數量上界 (單一攻擊行為通常 3-5 個階段)
HMM_COVARIANCE_TYPE = "diag"  # 共變異數類型: "diag" 或 "full"
HMM_MIN_COVAR = 0.1  # 最小共變異數 (防止特徵稀疏導致矩陣奇異)

# HMM 訓練與優化參數
HMM_N_STARTS = 4  # 隨機初始化次數 (控制耗時，仍保留多次嘗試)
HMM_N_ITER = 100  # Baum-Welch 最大迭代次數
HMM_TOL = 0.5  # 收斂閾值 (放寬以提早停)

# HMM 效能優化參數
HMM_ENABLE_PARALLEL = True  # 啟用並行 Grid Search
HMM_PARALLEL_N_JOBS = 4  # CPU 核心數 (-1 = 全部)
HMM_PARALLEL_BACKEND = "loky"  # joblib 後端: "loky" 或 "threading"

# HMM 模型儲存路徑 (Per-Dataset 策略下，模型儲存於各資料集目錄)
HMM_MODEL_DIR_PER_DATASET = True  # 啟用 Per-Dataset 模型儲存
CLUSTER_RESULTS_DIR = os.path.join(DATA_DIR, "SequenceClusters")



# ==================== External Sources (Threat Intel) ====================

# Reference CSVs
MITRE_TECHNIQUES_CSV = os.path.join(REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_WithDB_V1.csv")
MITRE_CODE_TOKENS_CSV = os.path.join(REFERENCE_RESOURCES_DIR, "MitreCodeTokens_V1.csv")


# External knowledge datasets (vector-space artifacts)
MITRE_EXTERNAL_KNOWLEDGE_DIR = os.path.join(EXTERNAL_KNOWLEDGE_DIR, "MITRE_ATTACK")

# Embedding defaults for external sources
EXTERNAL_SOURCES_BERT_MODEL_NAME = BERT_MODEL_NAME
EXTERNAL_SOURCES_BERT_CACHE_DIR = BERT_CACHE_DIR
EXTERNAL_SOURCES_EMBED_BATCH_SIZE = 32
EXTERNAL_SOURCES_EMBED_NORMALIZE = True

# External fetchers/cache
EXTERNAL_SOURCES_CACHE_DIR = os.path.join(REFERENCE_RESOURCES_DIR, "cache")
FETCHER_REQUEST_TIMEOUT_SECONDS = 60
CODE_EXTRACTOR_INPUT_DIR = os.path.join(DATA_DIR, "mitre_data")
CODE_EXTRACTOR_OUTPUT_DIR = os.path.join(DATA_DIR, "cti_code_only")
CODE_EXTRACTOR_TIMEOUT_SECONDS = 15
CODE_EXTRACTOR_MAX_WORKERS = 10
CODE_EXTRACTOR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Results/output
RESULT_DIR = "result"

# 其他設定
SEED = 42
