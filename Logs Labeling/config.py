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

# 參數設定
BERT_MODEL_NAME = "sentice-bert"
CHUNK_MODEL_PATH = os.path.join(BERT_MODEL_DIR, "chunk_model.pt")
ISOLATION_FOREST_PARAMS = {
	"n_estimators": 100,
	"contamination": 0.05,
	"random_state": 42
}
NMF_COMPONENTS = 10
HMM_STATES = 5

# 其他設定
SEED = 42
