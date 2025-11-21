
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
from typing import List, Tuple, Optional

# 調整匯入路徑，確保能載入同專案上層的 config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
import importlib

# * 配置參數
LOG_INPUT_PATH = config.INPUT_LOGS_DIR
LOG_INTERMEDIATE_PATH = config.INTERMEDIATE_DATA_DIR
LOG_OUTPUT_PATH = config.PROCESSED_LOGS_DIR
ENABLE_PARSER = config.ENABLE_PARSER  # 是否解析原始事件
LOG_PARSER = config.DEFAULT_PARSER  # 可選: 'drain', 'spell', 'lenma' (小寫模組名稱)


class LogLoader:
    """使用插件式解析器模組載入並解析日誌檔案。"""
    
    def __init__(
        self, 
        input_dir: str = LOG_INPUT_PATH, 
        output_dir: str = LOG_OUTPUT_PATH,
        enable_parser: bool = ENABLE_PARSER,
        parser_name: str = LOG_PARSER,
        use_registry_parser: bool = True,
        parser_config: dict = None
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_parser = enable_parser
        self.parser_name = parser_name.lower() if parser_name else 'drain'
        self.use_registry_parser = use_registry_parser
        
        # * 步驟 1: 僅在啟用解析時初始化解析器
        if self.enable_parser:
            # * 步驟 1.1: 動態導入解析器模組
            try:
                # 優先從 preprocess 子套件導入
                parser_module = importlib.import_module(f'preprocess.{self.parser_name}')
            except (ImportError, ModuleNotFoundError):
                try:
                    # 備選: 直接導入 (從 preprocess 目錄執行時)
                    parser_module = importlib.import_module(self.parser_name)
                except (ImportError, ModuleNotFoundError):
                    print(f"警告: 無法導入解析器 '{parser_name}'，改用 'drain'。")
                    self.parser_name = 'drain'
                    try:
                        parser_module = importlib.import_module(f'preprocess.{self.parser_name}')
                    except (ImportError, ModuleNotFoundError):
                        parser_module = importlib.import_module(self.parser_name)
            
            # * 步驟 1.2: 根據模組中可用的類別初始化解析器
            parser_config = parser_config or {}
            standard_config = parser_config.get('standard', {'depth': 4, 'st': 0.5})
            registry_config = parser_config.get('registry', {'depth': 6, 'st': 0.5, 'registry_mode': True})
            
            # * 步驟 1.3: 建立標準解析器 (通常是 DrainParser 或類似的)
            if hasattr(parser_module, 'DrainParser'):
                self.standard_parser = parser_module.DrainParser(**standard_config)
            elif hasattr(parser_module, 'Parser'):
                self.standard_parser = parser_module.Parser(**standard_config)
            else:
                raise AttributeError(f"解析器模組 '{parser_name}' 必須包含 'DrainParser' 或 'Parser' 類別")
            
            # * 步驟 1.4: 若需要則建立註冊表解析器
            if use_registry_parser:
                if hasattr(parser_module, 'RegistryDrainParser'):
                    self.registry_parser = parser_module.RegistryDrainParser(**registry_config)
                elif hasattr(parser_module, 'RegistryParser'):
                    self.registry_parser = parser_module.RegistryParser(**registry_config)
                else:
                    # 若無註冊表專用解析器則回退為標準解析器
                    self.registry_parser = None
                    print(f"注意: 解析器 '{parser_name}' 無註冊表專用解析器，僅使用標準解析器")
            else:
                self.registry_parser = None
        else:
            self.standard_parser = None
            self.registry_parser = None
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """解析單行日誌或返回原始日誌。
        
        若解析器停用，返回空模板/參數與原始日誌。
        否則委託給解析器的 parse_log_row 方法。
        """
        # * 步驟 2.1: 若解析器停用，建立日誌訊息但不解析
        if not self.enable_parser:
            log_parts = []
            for col in columns:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                        log_parts.append(str(val).strip())
            log_message = " ".join(log_parts)
            return "", [], log_message
        
        # * 步驟 2.2: 判斷使用哪個解析器
        operation = row.get('Operation', '')
        
        if (self.use_registry_parser and 
            self.registry_parser and 
            self.registry_parser.is_registry_operation(operation)):
            # 使用註冊表解析器
            return self.registry_parser.parse_log_row(row, columns)
        else:
            # 使用標準解析器
            return self.standard_parser.parse_log_row(row, columns)
    
    def parse_file(self, file_path: str, columns: List[str]) -> pd.DataFrame:
        """解析單個 CSV 檔案並提取模板和參數。"""
        # * 步驟 3.1: 讀取 CSV 檔案
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"讀取檔案錯誤 {file_path}: {e}")
            return None
        
        file_name = os.path.basename(file_path).split('.')[0]
        
        # * 步驟 3.2: 若解析器停用，直接合併選定欄位
        if not self.enable_parser:
            log_ids = []
            concatenated_logs = []
            
            for idx, row in df.iterrows():
                log_ids.append(f"{file_name}_{idx}")
                
                # 合併選定的欄位
                log_parts = []
                for col in columns:
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                            log_parts.append(str(val).strip())
                concatenated_logs.append(" ".join(log_parts))
            
            # 建立簡化的 DataFrame（無模板和參數）
            result_df = pd.DataFrame({
                'LogID': log_ids,
                'ConcatenatedLog': concatenated_logs
            })
            return result_df
        
        # * 步驟 3.3: 若解析器啟用，解析每一行日誌
        log_ids = []
        templates = []
        parameters = []
        original_logs = []
        
        for idx, row in df.iterrows():
            template, params, log_msg = self.parse_log_row(row, columns)
            log_ids.append(f"{file_name}_{idx}")
            templates.append(template)
            parameters.append('|'.join(params) if params else "")
            original_logs.append(log_msg)
        
        # * 步驟 3.4: 建立包含模板和參數的 DataFrame
        result_df = pd.DataFrame({
            'LogID': log_ids,
            'Template': templates,
            'Parameters': parameters,
            'OriginalLog': original_logs
        })
        
        return result_df
    
    def load_logs(
        self, 
        columns: List[str] = None,
        **kwargs
    ):
        """
        載入並解析日誌檔案。
        
        Args:
            columns: 要解析的欄位名稱列表 (預設: Operation, Path, Result, Command Line)
            num: 要處理的檔案數量
            ratio: 要處理的檔案比例 (0-1)
        """
        # * 步驟 4.1: 設定要解析的欄位
        if columns is None:
            columns = ["Operation", "Path", "Result", "Command Line"]
        
        # * 步驟 4.2: 確定檔案選擇範圍
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        
        if num:
            print(f"載入 {num} 個日誌檔案...")
        elif ratio:
            print(f"載入 {ratio*100:.1f}% 的日誌檔案...")
        else:
            print("載入所有日誌檔案...")
        
        # * 步驟 4.3: 獲取檔案列表
        all_files = [f for f in os.listdir(self.input_dir) if f.endswith(".csv")]
        
        if num:
            files = all_files[:num]
        elif ratio:
            files = all_files[:int(len(all_files) * ratio)]
        else:
            files = all_files
        
        # * 步驟 4.4: 建立輸出目錄
        os.makedirs(LOG_INTERMEDIATE_PATH, exist_ok=True)
        
        # * 步驟 4.5: 處理檔案
        parsed_dfs = []
        
        with tqdm(files, desc="解析日誌中", unit="檔案", dynamic_ncols=True) as pbar:
            for file in pbar:
                file_path = os.path.join(self.input_dir, file)
                short_name = file if len(file) <= 40 else file[:37] + "..."
                pbar.set_postfix({"解析器": self.parser_name, "檔案": short_name}, refresh=False)
                
                # * 步驟 4.6: 解析單個檔案
                parsed_df = self.parse_file(file_path, columns)
                
                if parsed_df is not None:
                    # * 步驟 4.7: 儲存解析結果
                    output_path = os.path.join(LOG_INTERMEDIATE_PATH, file)
                    parsed_df.to_csv(output_path, index=False, encoding='utf-8')
                    parsed_dfs.append(parsed_df)
        
        # * 步驟 4.8: 顯示處理統計資訊
        print(f"\n解析完成！")
        print(f"  已處理: {len(parsed_dfs)}/{len(files)} 個檔案")
        print(f"  輸出位置: {LOG_INTERMEDIATE_PATH}")
        
        if self.enable_parser:
            print(f"  解析器: {self.parser_name}")
            print(f"  標準模板數量: {len(self.standard_parser.get_clusters())}")
            if self.use_registry_parser and self.registry_parser:
                print(f"  註冊表模板數量: {len(self.registry_parser.get_clusters())}")
        else:
            print(f"  解析器: 已停用 (保留原始日誌)")
        
        return parsed_dfs


def main():
    # 1: 啟用解析 (預設)
    loader = LogLoader(enable_parser=True)
    loader.load_logs()
    
    # 2: 不解析 (保留原始日誌)
    # loader = LogLoader(enable_parser=False)
    # loader.load_logs(ratio=0.3)

if __name__ == "__main__":
    main()