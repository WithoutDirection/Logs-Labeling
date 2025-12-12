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
NMF_COMPONENTS = 10
HMM_STATES = 5

# 其他設定
SEED = 42
