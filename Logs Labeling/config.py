import os
from pathlib import Path

# Path resolution notes:
# - This project is often executed from different working directories.
# - Use paths relative to the repository root / package directory instead of CWD.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent

# ==================== 路徑設定（共用基礎） ====================
DATA_DIR = str(_REPO_ROOT / "data")
INPUT_LOGS_DIR = os.path.join(DATA_DIR, "input_logs")
REFERENCE_RESOURCES_DIR = os.path.join(DATA_DIR, "reference_resources")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "Intermediate_data")
PROCESSED_LOGS_DIR = os.path.join(DATA_DIR, "processed_logs")
BERT_MODEL_DIR = str(_PKG_DIR / "models" / "bert_model")
UNSUPERVISED_MODEL_DIR = str(_PKG_DIR / "models" / "Unsupervised")
HMM_MODEL_DIR = str(_PKG_DIR / "models" / "HMM")
RESULT_DIR = str(_REPO_ROOT / "result")  # 總輸出根目錄

# ==================== 預處理階段 ====================
ENABLE_PARSER = True  # 是否對原始事件進行樣板解析（False 則保留原字串）
PARSER_LIST = ["drain", "spell", "lenma"]  # 可用的解析器模組名稱（小寫）
DEFAULT_PARSER = "drain"  # 預設採用的解析器
ZIPF_PERCENTILE = 0.05  # 依 Zipf 法則移除高頻詞的百分位（前 5%）
USE_LOG_HIGH_FREQ = True  # 是否保留常見日誌詞彙（如 Authority、System 等）

# ==================== 嵌入與語言模型設定 ====================
# 可選：'sentence-bert', 'sentence-bert-multilingual', 'sentence-bert-large',
#      'bert-base-nli', 'bert-base', 'distilbert', 'roberta'，或自訂 HuggingFace 模型名稱
BERT_MODEL_NAME = "sentence-bert"
BERT_CACHE_DIR = os.path.join(BERT_MODEL_DIR, "cache")  # 模型快取路徑
BERT_AUTO_LOAD = True  # 是否在初始化時自動載入模型
CHUNK_MODEL_PATH = os.path.join(BERT_MODEL_DIR, "chunk_model.pt")

# ==================== 序列切分（Log Chunker） ====================
SEQUENCE_WINDOW_SIZE = 5  # 序列視窗大小
SEQUENCE_STRIDE = 3  # 視窗滑動步長（重疊量 = window_size - stride）
BILSTM_HIDDEN_SIZE = 128  # BiLSTM 隱藏層維度
BILSTM_NUM_LAYERS = 2  # BiLSTM 堆疊層數
BILSTM_DROPOUT = 0.3  # Dropout 比例
FUSION_ENABLE = False  # 是否啟用融合層
FUSION_OUTPUT_DIM = 256  # 融合層輸出維度
# LOG_VECTORS_DIR = os.path.join(DATA_DIR, "LogVectors")  # 若使用 Chunker，儲存 log vectors
LOG_VECTORS_DIR = os.path.join(DATA_DIR, "Embeddings")  # 若直接使用嵌入，讀取此路徑

# ==================== 概念抽取（NMF / LDA） ====================
# --- 模型參數 ---
NMF_COMPONENTS = 30  # 概念數量（潛在空間維度）
NMF_L1_REG = 0.01  # L1 正則化強度（控制稀疏度）
NMF_MAX_ITER = 500  # 最大迭代次數
NMF_TOL = 1e-3  # 收斂容許誤差
NMF_INIT = "nndsvd"  # 初始化方法（nndsvd 為確定性初始化，可確保結果可重現）
CONCEPT_SAMPLE_RATIO = 1.0  # 每個資料集用於訓練的樣本比例

# --- GPU 加速設定 ---
NMF_USE_GPU = True  # 是否優先使用 GPU 加速 NMF（若 CUDA 可用）
NMF_GPU_BATCH_SIZE = None  # GPU 訓練的 batch size（None = 自動根據 GPU 記憶體決定）
NMF_GPU_EPSILON = 1e-8  # 數值穩定性常數，防止除以零
NMF_GPU_CHECK_INTERVAL = 10  # 檢查收斂的間隔（每 N 次迭代）
NMF_GPU_VERBOSE = True  # 是否顯示 GPU 訓練進度

# --- 批次轉換並行設定（用於多資料集轉換，非 GPU 相關） ---
CONCEPT_BATCH_N_JOBS = -1  # 批次轉換並行數（-1 = 所有 CPU，1 = 無平行）

# --- 路徑設定 ---
NMF_MODEL_PATH = str(_PKG_DIR / "models" / "nmf_concept_model.pkl")  # 概念模型儲存路徑
CONCEPT_VECTORS_DIR = os.path.join(DATA_DIR, "ConceptVectors")  # 概念向量輸出目錄
EXTERNAL_KNOWLEDGE_DIR = os.path.join(DATA_DIR, "ExternalKnowledge")  # 外部知識向量根目錄

# ==================== 異常偵測（Unsupervised） ====================
# Isolation Forest
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.05  # 預期異常比例，可設為 "auto"
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42

# COPOD
COPOD_CONTAMINATION = 0.05

# AutoEncoder
AE_LATENT_DIM = 32
AE_HIDDEN_DIMS = [128, 64]
AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3

# PCA + GMM
PCA_EXPLAINED_VAR = 0.95
GMM_N_COMPONENTS_RANGE = (2, 10)
GMM_COVARIANCE_TYPE = "full"
GMM_USE_BIC = True

# 集成與閾值設定
DETECTION_MODELS = ["isolation_forest", "copod", "autoencoder", "pca_gmm"]
SCORE_SCALER = "minmax"  # 可選: "minmax", "rank", "zscore"
THRESHOLDING_METHOD = "mad"  # 可選: "percentile", "std", "top_n", "mad"
THRESHOLDING_PARAMS = {"percentile": 98}  # 對應閾值方法的參數
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.25,
    "copod": 0.25,
    "autoencoder": 0.25,
    "pca_gmm": 0.25,
}
DETECTION_RESULTS_DIR = os.path.join(DATA_DIR, "Detection_Results")
DETECTION_VIZ_DIR = os.path.join(RESULT_DIR, "Anomaly_Detection")

# MAD (Median Absolute Deviation) 自適應閾值設定
MAD_THRESHOLD_MULTIPLIER = 3.0  # MAD 乘數，通常使用 2.5-3.5
MAD_USE_MODIFIED = True  # 是否讓 MAD 更接近標準差

# 時間序列平滑化設定
ENABLE_TIME_SERIES_SMOOTHING = True  # 是否啟用時間序列平滑化
SMOOTHING_WINDOW_SIZE = 5  # 滑動視窗大小（建議為奇數）
SMOOTHING_METHOD = "mean"  # 可選: "mean", "median", "gaussian"

# 模型相關性分析設定
ENABLE_CORRELATION_ANALYSIS = True  # 是否啟用相關性分析
CORRELATION_METHOD = "pearson"  # 可選: "pearson", "spearman", "kendall"


# ==================== 序列分群（HMM；每資料集分開） ====================
HMM_STATES = 3  # 基本狀態數（若未啟用搜尋時）
HMM_K_MIN = 1  # 隱藏狀態下界
HMM_K_MAX = 10  # 隱藏狀態上界（常見攻擊序列約 3-5 階段）
HMM_COVARIANCE_TYPE = "diag"  # 共變異數型式: "diag" 或 "full"
HMM_MIN_COVAR = 0.1  # 最小共變異數，避免矩陣奇異
HMM_N_STARTS = 4  # 隨機初始化次數（平衡耗時與穩定性）
HMM_N_ITER = 100  # Baum-Welch 最大迭代
HMM_TOL = 0.5  # 收斂門檻
HMM_ENABLE_PARALLEL = True  # 是否平行搜尋
HMM_PARALLEL_N_JOBS = 4  # 平行核心數（-1 代表全部）
HMM_PARALLEL_BACKEND = "loky"  # joblib 後端
HMM_MODEL_DIR_PER_DATASET = True  # 每資料集獨立儲存模型
CLUSTER_RESULTS_DIR = os.path.join(DATA_DIR, "SequenceClusters")

# ==================== 外部威脅情報與代碼抽取 ====================
MITRE_TECHNIQUES_CSV = os.path.join(REFERENCE_RESOURCES_DIR, "MitreTechniquesTokens_V5.csv")
MITRE_CODE_TOKENS_CSV = os.path.join(REFERENCE_RESOURCES_DIR, "MitreCodeTokens_V1.csv")
MITRE_EXTERNAL_KNOWLEDGE_DIR = os.path.join(EXTERNAL_KNOWLEDGE_DIR, "MITRE_RAW_EMBEDDINGS")
EXTERNAL_SOURCES_BERT_MODEL_NAME = BERT_MODEL_NAME
EXTERNAL_SOURCES_BERT_CACHE_DIR = BERT_CACHE_DIR
EXTERNAL_SOURCES_EMBED_BATCH_SIZE = 32
EXTERNAL_SOURCES_EMBED_NORMALIZE = True
EXTERNAL_SOURCES_CACHE_DIR = os.path.join(REFERENCE_RESOURCES_DIR, "cache")
FETCHER_REQUEST_TIMEOUT_SECONDS = 60
CODE_EXTRACTOR_INPUT_DIR = os.path.join(DATA_DIR, "mitre_data")
CODE_EXTRACTOR_OUTPUT_DIR = os.path.join(DATA_DIR, "cti_code_only")
CODE_EXTRACTOR_TIMEOUT_SECONDS = 15
CODE_EXTRACTOR_MAX_WORKERS = 10
CODE_EXTRACTOR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}

# ==================== 自動標註（Auto-Labeling） ====================
# 相似度閾值：低於此值的相似度將被視為不匹配
LABELING_SIMILARITY_THRESHOLD = 0.3

# 信心度閾值：similarity * confidence < threshold 則標記為 Benign
LABELING_CONFIDENCE_THRESHOLD = 0.2

# 異常分數權重（用於計算最終 confidence）
# confidence = anomaly_weight * anomaly_score + similarity_weight * similarity
LABELING_ANOMALY_WEIGHT = 0.3

# 相似度權重
LABELING_SIMILARITY_WEIGHT = 0.7

# Top-K 候選技術（用於輸出多個可能的匹配結果）
LABELING_TOP_K = 3

# 是否使用原始嵌入向量進行比對（True: Raw Embeddings, False: Concept Vectors）
LABELING_USE_RAW_EMBEDDINGS = False

# 混合評分設定 (Hybrid Scoring)
LABELING_USE_TFIDF = True           # 是否啟用 TF-IDF 輔助評分
LABELING_WEIGHT_EMBEDDING = 0.7     # 嵌入向量相似度權重
LABELING_WEIGHT_TFIDF = 0.3         # TF-IDF 關鍵字相似度權重
MITRE_TFIDF_DIR = os.path.join(EXTERNAL_KNOWLEDGE_DIR, "MITRE_TFIDF")

# 標註結果輸出目錄
LABELING_RESULTS_DIR = os.path.join(RESULT_DIR, "Labeling_Results")

# ==================== 結果與通用設定 ====================

SEED = 42  # 全域隨機種子
