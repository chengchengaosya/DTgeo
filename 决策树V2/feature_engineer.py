"""
feature_engineer.py
特征工程：分箱 + 交互 + 穷举子集选择
所有参数从 config.py 读取
"""

import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import f1_score
import config


def create_binned_features(X_raw, feature_names):
    """
    多粒度分箱：将连续特征转为离散特征
    地质特征存在阈值效应，分箱能帮助决策树更快找到有效分裂点
    """
    print("\n[...] 特征分箱...")
    new_features = []
    new_names = []

    col_idx = {name: i for i, name in enumerate(feature_names)}

    # SGR 分箱
    if 'SGR' in col_idx:
        sgr = X_raw[:, col_idx['SGR']]
        sgr_binned = np.digitize(sgr, config.SGR_BINS[1:-1]).astype(float)
        new_features.append(sgr_binned)
        new_names.append('SGR_bin')

    # 碳酸盐岩 分箱
    if '碳酸盐岩' in col_idx:
        ca = X_raw[:, col_idx['碳酸盐岩']]
        ca_binned = np.digitize(ca, config.CARBONATE_BINS[1:-1]).astype(float) \
            if hasattr(config, 'CARBONATE_BINS') else \
            np.digitize(ca, config.ITE_BINS[1:-1]).astype(float)
        new_features.append(ca_binned)
        new_names.append('碳酸盐岩_bin')

    # 倾角 分箱
    if '倾角中值' in col_idx:
        angle = X_raw[:, col_idx['倾角中值']]
        angle_binned = np.digitize(angle, config.ANGLE_BINS[1:-1]).astype(float)
        new_features.append(angle_binned)
        new_names.append('倾角_bin')

    # 断距 等分位分箱（3分位）
    if '断距中值' in col_idx:
        fault = X_raw[:, col_idx['断距中值']]
        q33, q66 = np.percentile(fault, [33, 66])
        fault_binned = np.digitize(fault, [q33, q66]).astype(float)
        new_features.append(fault_binned)
        new_names.append('断距_bin')

    # 微裂缝 二值化
    if '微裂缝' in col_idx:
        micro = X_raw[:, col_idx['微裂缝']]
        micro_binary = (micro > 0).astype(float)
        new_features.append(micro_binary)
        new_names.append('有微裂缝')

    if new_features:
        X_binned = np.column_stack(new_features)
        print(f"    生成 {len(new_names)} 个分箱特征：{new_names}")
    else:
        X_binned = np.empty((X_raw.shape[0], 0))

    return X_binned, new_names


def create_interaction_features(X_raw, feature_names):
    """交互特征：两两乘积 + 关键比值"""
    if not config.USE_INTERACTION_FEATURES:
        return np.empty((X_raw.shape[0], 0)), []

    print("[...] 交互特征...")
    col_idx = {name: i for i, name in enumerate(feature_names)}
    new_features = []
    new_names = []
    eps = 1e-8

    # 预定义的有地质意义的交互
    interactions = [
        ('SGR', '碳酸盐岩', 'SGR×碳酸盐岩', 'mul'),
        ('SGR', '断距中值',  'SGR×断距',     'mul'),
        ('断距中值', '倾角中值', '断距×倾角',  'mul'),
        ('断距中值', '延伸长度(km)', '断距/延伸', 'div'),
        ('碳酸盐岩', 'SGR',   '碳酸盐岩/SGR', 'div'),
    ]

    for f1, f2, name, op in interactions:
        if f1 in col_idx and f2 in col_idx:
            v1 = X_raw[:, col_idx[f1]]
            v2 = X_raw[:, col_idx[f2]]
            if op == 'mul':
                new_features.append(v1 * v2)
            else:
                new_features.append(v1 / (v2 + eps))
            new_names.append(name)

    if new_features:
        X_inter = np.column_stack(new_features)
        print(f"    生成 {len(new_names)} 个交互特征：{new_names}")
    else:
        X_inter = np.empty((X_raw.shape[0], 0))

    return X_inter, new_names


def build_all_features(X_raw, base_names):
    """合并原始 + 分箱 + 交互特征"""
    X_bin, bin_names = create_binned_features(X_raw, base_names)
    X_inter, inter_names = create_interaction_features(X_raw, base_names)

    parts = [X_raw]
    names = list(base_names)
    if X_bin.shape[1] > 0:
        parts.append(X_bin)
        names.extend(bin_names)
    if X_inter.shape[1] > 0:
        parts.append(X_inter)
        names.extend(inter_names)

    X_all = np.column_stack(parts) if len(parts) > 1 else X_raw.copy()
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[✓] 全部候选特征：{len(names)} 个")
    return X_all, names


def select_best_features(X_all, y, all_names):
    """
    穷举特征子集，用 LOOCV + 决策树 F1 选最优组合
    这是小样本下最可靠的特征选择方式
    """
    print("\n" + "=" * 55)
    print("          🔍 穷举特征子集（LOOCV）")
    print("=" * 55)

    n_features = len(all_names)
    loo = LeaveOneOut()
    max_size = min(config.MAX_SUBSET_SIZE, n_features)

    best_score = 0
    best_indices = None
    best_size = 0
    total_searched = 0

    for size in range(2, max_size + 1):
        combos = list(itertools.combinations(range(n_features), size))

        if len(combos) > config.MAX_COMBOS_PER_SIZE:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(combos), config.MAX_COMBOS_PER_SIZE, replace=False)
            combos = [combos[i] for i in idx]

        for indices in combos:
            total_searched += 1
            X_sub = X_all[:, indices]
            clf = DecisionTreeClassifier(max_depth=None, class_weight='balanced',
                                         random_state=42)
            y_pred = cross_val_predict(clf, X_sub, y, cv=loo)
            score = f1_score(y, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_indices = list(indices)
                best_size = size

        print(f"    size={size}: 已搜索 {total_searched} 组, 当前最优 F1={best_score:.4f}")

    selected_names = [all_names[i] for i in best_indices]
    print(f"\n[✓] 最优子集（{best_size}个）LOOCV F1 = {best_score:.4f}")
    print(f"    特征：{selected_names}")

    return best_indices, selected_names, best_score