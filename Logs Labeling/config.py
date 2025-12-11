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

ISOLATION_FOREST_PARAMS = {
	"n_estimators": 100,
	"contamination": 0.05,
	"random_state": 42
}
NMF_COMPONENTS = 10
HMM_STATES = 5

# 其他設定
SEED = 42
