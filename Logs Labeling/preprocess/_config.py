"""
preprocess 模組共用配置

此模組提供 preprocess 子套件的共用配置與路徑設定。
"""
import sys
from pathlib import Path

# 調整匯入路徑，確保能載入專案根目錄的 config.py
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

# ======================== 路徑配置 ========================
LOG_INPUT_PATH = config.INPUT_LOGS_DIR
LOG_INTERMEDIATE_PATH = config.INTERMEDIATE_DATA_DIR
LOG_OUTPUT_PATH = config.PROCESSED_LOGS_DIR

# ======================== 解析器配置 ========================
ENABLE_PARSER = config.ENABLE_PARSER
LOG_PARSER = config.DEFAULT_PARSER  # 'drain', 'spell', 'lenma'

# ======================== 嵌入配置 ========================
BERT_MODEL_NAME = config.BERT_MODEL_NAME
BERT_CACHE_DIR = config.BERT_CACHE_DIR
DATA_DIR = config.DATA_DIR

# ======================== BiLSTM 配置 ========================
LOG_VECTORS_DIR = config.LOG_VECTORS_DIR
SEQUENCE_WINDOW_SIZE = config.SEQUENCE_WINDOW_SIZE
SEQUENCE_STRIDE = config.SEQUENCE_STRIDE
BILSTM_HIDDEN_SIZE = config.BILSTM_HIDDEN_SIZE
BILSTM_NUM_LAYERS = config.BILSTM_NUM_LAYERS
BILSTM_DROPOUT = config.BILSTM_DROPOUT
FUSION_ENABLE = config.FUSION_ENABLE
FUSION_OUTPUT_DIM = config.FUSION_OUTPUT_DIM
