#!/usr/bin/env python3
"""
資料目錄清理工具

此腳本用於清理管線執行過程中產生的中間資料和輸出結果，
但會保留輸入資料（input_logs）和參考資源（reference_resources）。

使用方式:
    # 預覽將要刪除的目錄（不實際刪除）
    python clear_data.py

    # 實際執行清理
    python clear_data.py --execute

    # 自訂要保留的目錄
    python clear_data.py --execute --preserve input_logs reference_resources my_backup
"""

import sys
import os
import argparse

# 確保能夠導入 utils 模組
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.path import clear_data_directories
import config


def main():
    parser = argparse.ArgumentParser(
        description="清理資料目錄中的中間資料和輸出結果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 預覽將要刪除的目錄
  python clear_data.py

  # 實際執行清理
  python clear_data.py --execute

  # 自訂要保留的目錄
  python clear_data.py --execute --preserve input_logs reference_resources
        """
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="實際執行刪除操作（預設為預覽模式）"
    )
    
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help=f"資料根目錄路徑（預設: {config.DATA_DIR}）"
    )
    
    parser.add_argument(
        "--preserve",
        nargs="*",
        default=None,
        help="要保留的子目錄名稱（預設: input_logs reference_resources）"
    )
    
    args = parser.parse_args()
    
    # 處理保留目錄
    preserve_dirs = None
    if args.preserve is not None:
        if len(args.preserve) == 0:
            print("[錯誤] --preserve 參數至少需要指定一個目錄")
            sys.exit(1)
        preserve_dirs = set(args.preserve)
    
    # 顯示警告訊息
    if args.execute:
        print("\n⚠️  警告: 即將刪除資料目錄中的檔案！")
        if preserve_dirs:
            print(f"保留目錄: {', '.join(sorted(preserve_dirs))}")
        else:
            print("保留目錄: input_logs, reference_resources")
        
        response = input("\n確定要繼續嗎？(yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("操作已取消")
            sys.exit(0)
    
    # 執行清理
    result = clear_data_directories(
        data_dir=args.data_dir,
        preserve_dirs=preserve_dirs,
        dry_run=not args.execute
    )
    
    # 顯示結果摘要
    if args.execute and result["removed"]:
        print("\n✓ 清理完成")
    elif not args.execute:
        print("\n💡 使用 --execute 參數來實際執行刪除操作")


if __name__ == "__main__":
    main()
