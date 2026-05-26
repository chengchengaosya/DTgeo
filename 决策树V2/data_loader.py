"""
data_loader.py
数据读取与校验（纯净，不做特征工程）
"""

import pandas as pd
import numpy as np
import sys
import config


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_excel(config.DATA_PATH, engine='openpyxl')
        print(f"[✓] 读取数据：{df.shape[0]} 行 × {df.shape[1]} 列")
        return df
    except FileNotFoundError:
        print(f"[✗] 文件未找到：{config.DATA_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"[✗] 读取失败：{e}")
        sys.exit(1)


def validate_columns(df: pd.DataFrame) -> None:
    required = config.BASE_FEATURES + [config.TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[✗] 缺少列：{missing}")
        sys.exit(1)
    print(f"[✓] 所需列完整")


def validate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    check_cols = config.BASE_FEATURES + [config.TARGET_COL]
    invalid_summary = {}
    for col in check_cols:
        original = df[col].copy()
        converted = pd.to_numeric(df[col], errors='coerce')
        mask = converted.isna() & original.notna()
        if mask.any():
            invalid_summary[col] = mask.sum()
        df.loc[:, col] = converted

    if invalid_summary:
        print(f"[⚠] 非法值：{invalid_summary}")

    before = len(df)
    df = df.dropna(subset=check_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[⚠] 删除 {dropped} 行缺失值，剩余 {len(df)} 行")
    else:
        print(f"[✓] 数值校验通过，{len(df)} 行")
    return df


def validate_target(df: pd.DataFrame) -> None:
    vals = set(df[config.TARGET_COL].unique())
    if not vals.issubset({0, 1, 0.0, 1.0}):
        print(f"[✗] 目标列包含非 0/1 值：{vals - {0,1,0.0,1.0}}")
        sys.exit(1)
    counts = df[config.TARGET_COL].value_counts()
    print(f"[✓] 目标分布：0→{counts.get(0,0)}, 1→{counts.get(1,0)}")


def prepare_raw_data():
    """加载原始数据，只做校验不做特征工程"""
    print("=" * 55)
    print("          📂 数据加载")
    print("=" * 55)
    df = load_data()
    validate_columns(df)
    df = validate_numeric(df)
    validate_target(df)

    for col in config.BASE_FEATURES:
        df[col] = df[col].astype(np.float64)

    X_raw = df[config.BASE_FEATURES].values
    y = df[config.TARGET_COL].astype(int).values
    print(f"[✓] 原始特征矩阵：{X_raw.shape}")
    print("=" * 55)
    return X_raw, y, df