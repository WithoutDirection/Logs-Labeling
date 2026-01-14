"""
LogLoader：日誌載入與解析模組

此模組負責：
1. 載入原始日誌 CSV 檔案
2. 可選：使用 Drain 等解析器提取模板與參數
3. 輸出中間格式供後續嵌入使用
"""
import pandas as pd
import importlib
from tqdm import tqdm
from typing import List, Tuple, Optional

from ._config import (
    LOG_INPUT_PATH, LOG_INTERMEDIATE_PATH, LOG_OUTPUT_PATH,
    ENABLE_PARSER, LOG_PARSER
)
from utils.path import join_path, get_stem, ensure_dir, get_filtered_files


class LogLoader:
    """
    使用插件式解析器模組載入並解析日誌檔案。
    
    支援多種解析器（Drain, Spell, Lenma），可透過 parser_name 參數切換。
    若 enable_parser=False，則保留原始日誌不解析。
    
    Example:
        >>> loader = LogLoader(enable_parser=False)
        >>> dfs = loader.load_logs(num=10)
    """
    
    def __init__(
        self, 
        input_dir: str = LOG_INPUT_PATH, 
        output_dir: str = LOG_OUTPUT_PATH,
        enable_parser: bool = ENABLE_PARSER,
        parser_name: str = LOG_PARSER,
        use_registry_parser: bool = True,
        parser_config: dict = None
    ):
        """
        初始化 LogLoader。
        
        Args:
            input_dir: 輸入日誌目錄
            output_dir: 輸出目錄
            enable_parser: 是否啟用解析器
            parser_name: 解析器名稱（drain/spell/lenma）
            use_registry_parser: 是否使用專用的註冊表解析器
            parser_config: 解析器配置字典
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_parser = enable_parser
        self.parser_name = parser_name.lower() if parser_name else 'drain'
        self.use_registry_parser = use_registry_parser
        
        self.standard_parser = None
        self.registry_parser = None
        
        if self.enable_parser:
            self._init_parsers(parser_config)
    
    def _init_parsers(self, parser_config: dict = None):
        """初始化解析器。"""
        # 動態導入解析器模組
        parser_module = self._load_parser_module()
        
        parser_config = parser_config or {}
        standard_config = parser_config.get('standard', {'depth': 4, 'st': 0.5})
        registry_config = parser_config.get('registry', {'depth': 6, 'st': 0.5, 'registry_mode': True})
        
        # 建立標準解析器
        if hasattr(parser_module, 'DrainParser'):
            self.standard_parser = parser_module.DrainParser(**standard_config)
        elif hasattr(parser_module, 'Parser'):
            self.standard_parser = parser_module.Parser(**standard_config)
        else:
            raise AttributeError(
                f"解析器模組 '{self.parser_name}' 必須包含 'DrainParser' 或 'Parser' 類別"
            )
        
        # 建立註冊表解析器（可選）
        if self.use_registry_parser:
            if hasattr(parser_module, 'RegistryDrainParser'):
                self.registry_parser = parser_module.RegistryDrainParser(**registry_config)
            elif hasattr(parser_module, 'RegistryParser'):
                self.registry_parser = parser_module.RegistryParser(**registry_config)
            else:
                self.registry_parser = None
                print(f"[Info] 解析器 '{self.parser_name}' 無註冊表專用解析器")
    
    def _load_parser_module(self):
        """動態載入解析器模組。"""
        try:
            return importlib.import_module(f'preprocess.{self.parser_name}')
        except (ImportError, ModuleNotFoundError):
            try:
                return importlib.import_module(self.parser_name)
            except (ImportError, ModuleNotFoundError):
                print(f"[Warning] 無法導入解析器 '{self.parser_name}'，改用 'drain'")
                self.parser_name = 'drain'
                try:
                    return importlib.import_module(f'preprocess.{self.parser_name}')
                except (ImportError, ModuleNotFoundError):
                    return importlib.import_module(self.parser_name)
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """
        解析單行日誌。
        
        Returns:
            (template, parameters, original_log)
        """
        if not self.enable_parser:
            log_parts = []
            for col in columns:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                        log_parts.append(str(val).strip())
            return "", [], " ".join(log_parts)
        
        operation = row.get('Operation', '')
        
        if (self.use_registry_parser and 
            self.registry_parser and 
            self.registry_parser.is_registry_operation(operation)):
            return self.registry_parser.parse_log_row(row, columns)
        else:
            return self.standard_parser.parse_log_row(row, columns)
    
    def parse_file(self, file_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
        """解析單個 CSV 檔案。"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"[Error] 讀取檔案失敗 {file_path}: {e}")
            return None
        
        file_name = get_stem(file_path)
        
        if not self.enable_parser:
            # 無解析：合併欄位
            log_ids = []
            concatenated_logs = []
            
            for idx, row in df.iterrows():
                log_ids.append(f"{file_name}_{idx}")
                log_parts = []
                for col in columns:
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                            log_parts.append(str(val).strip())
                concatenated_logs.append(" ".join(log_parts))
            
            return pd.DataFrame({
                'LogID': log_ids,
                'ConcatenatedLog': concatenated_logs
            })
        
        # 有解析：提取模板與參數
        log_ids, templates, parameters, original_logs = [], [], [], []
        
        for idx, row in df.iterrows():
            template, params, log_msg = self.parse_log_row(row, columns)
            log_ids.append(f"{file_name}_{idx}")
            templates.append(template)
            parameters.append('|'.join(params) if params else "")
            original_logs.append(log_msg)
        
        return pd.DataFrame({
            'LogID': log_ids,
            'Template': templates,
            'Parameters': parameters,
            'OriginalLog': original_logs
        })
    
    def load_logs(self, columns: List[str] = None, **kwargs) -> List[pd.DataFrame]:
        """
        載入並解析日誌檔案。
        
        Args:
            columns: 要解析的欄位（預設: Operation, Path, Result, Command Line）
            num: 要處理的檔案數量
            ratio: 要處理的檔案比例 (0-1)
            
        Returns:
            解析後的 DataFrame 列表
        """
        if columns is None:
            columns = ["Operation", "Path", "Result", "Command Line"]
        
        num = kwargs.get("num")
        ratio = kwargs.get("ratio")
        files = get_filtered_files(self.input_dir, ".csv", num=num, ratio=ratio)
        
        # 顯示載入資訊
        if num:
            print(f"載入 {num} 個日誌檔案...")
        elif ratio:
            print(f"載入 {ratio*100:.1f}% 的日誌檔案...")
        else:
            print("載入所有日誌檔案...")
        
        ensure_dir(LOG_INTERMEDIATE_PATH)
        
        parsed_dfs = []
        with tqdm(files, desc="解析日誌中", unit="檔案", dynamic_ncols=True) as pbar:
            for file in pbar:
                file_path = join_path(self.input_dir, file)
                short_name = file if len(file) <= 40 else file[:37] + "..."
                pbar.set_postfix({"解析器": self.parser_name, "檔案": short_name}, refresh=False)
                
                parsed_df = self.parse_file(file_path, columns)
                if parsed_df is not None:
                    output_path = join_path(LOG_INTERMEDIATE_PATH, file)
                    parsed_df.to_csv(output_path, index=False, encoding='utf-8')
                    parsed_dfs.append(parsed_df)
        
        # 顯示統計
        print(f"\n解析完成！")
        print(f"  已處理: {len(parsed_dfs)}/{len(files)} 個檔案")
        print(f"  輸出位置: {LOG_INTERMEDIATE_PATH}")
        
        if self.enable_parser:
            print(f"  解析器: {self.parser_name}")
            print(f"  標準模板數量: {len(self.standard_parser.get_clusters())}")
            if self.registry_parser:
                print(f"  註冊表模板數量: {len(self.registry_parser.get_clusters())}")
        else:
            print(f"  解析器: 已停用 (保留原始日誌)")
        
        return parsed_dfs
