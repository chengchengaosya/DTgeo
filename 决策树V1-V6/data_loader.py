"""
data_loader.py  v6.0
数据读取 + 校验 + 精简特征工程
核心：不删除任何数据，保留全部183条
"""

import pandas as pd
import numpy as np
import sys

BASE_FEATURE_COLS = ['断距中值', '倾角中值', '延伸长度(km)', 'SGR', '碳酸盐岩', '微裂缝']
TARGET_COL = '断层启闭性赋值'


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        print(f"[✓] 成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
        return df
    except FileNotFoundError:
        print(f"[✗] 文件未找到：{filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[✗] 读取文件失败：{e}")
        sys.exit(1)


def validate_columns(df: pd.DataFrame) -> None:
    required = BASE_FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[✗] 缺少列：{missing}")
        sys.exit(1)
    print(f"[✓] 所需列均存在")


def validate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    check_cols = BASE_FEATURE_COLS + [TARGET_COL]
    invalid_summary = {}
    for col in check_cols:
        original = df[col].copy()
        converted = pd.to_numeric(df[col], errors='coerce')
        mask = converted.isna() & original.notna()
        if mask.any():
            bad_indices = df.index[mask].tolist()
            bad_values = original[mask].tolist()
            invalid_summary[col] = list(zip(bad_indices, bad_values))
        df.loc[:, col] = converted

    if invalid_summary:
        print("\n[⚠] 非法值：")
        for col, records in invalid_summary.items():
            print(f"    列「{col}」：{len(records)} 处")

    before = len(df)
    df = df.dropna(subset=check_cols).reset_index(drop=True)
    after = len(df)
    if before - after > 0:
        print(f"[⚠] 删除 {before - after} 行缺失值，剩余 {after} 行")
    else:
        print(f"[✓] 数值校验通过，共 {after} 行")
    return df


def validate_target(df: pd.DataFrame) -> None:
    unique_vals = set(df[TARGET_COL].unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        print(f"[✗] 目标列包含非 0/1 的值")
        sys.exit(1)
    counts = df[TARGET_COL].value_counts()
    ratio = counts.min() / counts.max()
    print(f"[✓] 目标分布：\n{counts.to_string()}")
    print(f"    类别比例：{ratio:.3f}")


def feature_engineering(df: pd.DataFrame):
    """
    v6 特征工程：只添加少量高质量特征
    原则：特征数 << 样本数的平方根（√183 ≈ 13.5）
    """
    print("\n[...] 特征工程...")

    for col in BASE_FEATURE_COLS:
        df[col] = df[col].astype(np.float64)

    eps = 1e-8
    # 仅保留 3 个最有地质意义的衍生特征
    df['SGR×碳酸盐岩'] = df['SGR'].values * df['碳酸盐岩'].values
    df['SGR×断距']     = df['SGR'].values * df['断距中值'].values
    df['断距/延伸']    = df['断距中值'].values / (df['延伸长度(km)'].values + eps)

    feature_cols = BASE_FEATURE_COLS + ['SGR×碳酸盐岩', 'SGR×断距', '断距/延伸']

    print(f"[✓] 特征数：{len(feature_cols)}（原始6 + 衍生3）")
    return df, feature_cols


def prepare_data(filepath: str):
    print("=" * 55)
    print("          📂 数据加载与校验")
    print("=" * 55)

    df = load_data(filepath)
    validate_columns(df)
    df = validate_numeric(df)
    validate_target(df)
    # v6：不删除离群点，保留全部数据
    df, feature_cols = feature_engineering(df)

    X = df[feature_cols].values.astype(np.float64)
    y = df[TARGET_COL].astype(int).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[✓] X: {X.shape}，y: {y.shape}")
    print(f"    不删除任何数据，最大化利用全部样本")
    print("=" * 55)
    return X, y, feature_cols, df