import sys
import os

print("✔ Python 位置：", sys.executable)
print("✔ 當前環境：", os.environ.get('VIRTUAL_ENV', '無虛擬環境'))
print("✔ 嘗試載入 pandas...")

try:
    import pandas as pd
    print("✅ pandas 載入成功，版本：", pd.__version__)
except ModuleNotFoundError:
    print("❌ 找不到 pandas")
