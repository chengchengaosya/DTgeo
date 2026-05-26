"""
bin_optimizer.py
分箱阈值预分析工具：独立运行，为每个特征寻找最优分箱方案
运行方式：python bin_optimizer.py
输出：推荐的分箱阈值 → 手动填入 config.py
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  基础配置（和 config.py 保持一致）
# ============================================================
DATA_PATH = '决策树—断层封闭性数据集.xlsx'
BASE_FEATURES = ['断距中值', '倾角中值', '延伸长度(km)', 'SGR', '碳酸盐岩', '微裂缝']
TARGET_COL = '断层启闭性赋值'
OUTPUT_DIR = 'bin_analysis'


# ============================================================
#  1. 数据加载
# ============================================================
def load_data():
    """加载并校验数据"""
    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=BASE_FEATURES + [TARGET_COL]).reset_index(drop=True)

    for col in BASE_FEATURES:
        df[col] = df[col].astype(np.float64)

    X = df[BASE_FEATURES].values
    y = df[TARGET_COL].astype(int).values
    print(f"[✓] 加载数据：{X.shape[0]} 条，{X.shape[1]} 个特征")
    print(f"    类别分布：0→{(y==0).sum()}, 1→{(y==1).sum()}")
    return X, y, df


# ============================================================
#  2. 单特征分布分析
# ============================================================
def analyze_feature_distribution(X, y, feature_names, save_dir):
    """
    分析每个特征在两个类别中的分布差异
    帮助理解哪些特征有区分力，以及分界点大概在哪
    """
    print("\n" + "=" * 60)
    print("  📊 特征分布分析")
    print("=" * 60)

    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        vals_0 = X[y == 0, i]
        vals_1 = X[y == 1, i]

        # 直方图
        ax.hist(vals_0, bins=20, alpha=0.5, color='steelblue', label='封闭(0)', density=True)
        ax.hist(vals_1, bins=20, alpha=0.5, color='darkorange', label='开启(1)', density=True)

        # 统计量
        mean_0, mean_1 = vals_0.mean(), vals_1.mean()
        med_0, med_1 = np.median(vals_0), np.median(vals_1)

        ax.axvline(mean_0, color='steelblue', ls='--', lw=1.5, label=f'均值0={mean_0:.2f}')
        ax.axvline(mean_1, color='darkorange', ls='--', lw=1.5, label=f'均值1={mean_1:.2f}')

        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=7)

        # 打印统计
        sep_ratio = abs(mean_0 - mean_1) / (max(vals_0.std(), vals_1.std()) + 1e-8)
        print(f"\n    [{name}]")
        print(f"      封闭(0): 均值={mean_0:.3f}, 中位数={med_0:.3f}, "
              f"范围=[{vals_0.min():.3f}, {vals_0.max():.3f}]")
        print(f"      开启(1): 均值={mean_1:.3f}, 中位数={med_1:.3f}, "
              f"范围=[{vals_1.min():.3f}, {vals_1.max():.3f}]")
        print(f"      类间分离度：{sep_ratio:.3f} {'⭐ 区分力强' if sep_ratio > 0.5 else ''}")

    plt.suptitle('各特征在两类中的分布 / Feature Distribution by Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
#  3. 单特征最优分箱搜索
# ============================================================
def find_best_bins_single_feature(X_col, y, n_bins_range=(2, 6)):
    """
    对单个特征穷举不同分箱方案，用 5 折 CV + 决策树评估
    分箱方案包括：等距、等频、基于百分位

    返回：最优分箱阈值、最优得分、所有方案的结果
    """
    results = []

    for n_bins in range(n_bins_range[0], n_bins_range[1] + 1):

        # ---- 方案 1：等频分箱（分位数）----
        try:
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            thresholds_quantile = np.percentile(X_col, percentiles)
            thresholds_quantile = np.unique(thresholds_quantile)  # 去重
            if len(thresholds_quantile) >= 1:
                X_binned = np.digitize(X_col, thresholds_quantile).reshape(-1, 1).astype(float)
                clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                y_pred = cross_val_predict(clf, X_binned, y, cv=skf)
                score = f1_score(y, y_pred, zero_division=0)
                results.append({
                    'method': f'等频{n_bins}箱',
                    'thresholds': thresholds_quantile.tolist(),
                    'n_bins': len(thresholds_quantile) + 1,
                    'f1': score
                })
        except Exception:
            pass

        # ---- 方案 2：等距分箱 ----
        try:
            vmin, vmax = X_col.min(), X_col.max()
            thresholds_uniform = np.linspace(vmin, vmax, n_bins + 1)[1:-1]
            thresholds_uniform = np.unique(thresholds_uniform)
            if len(thresholds_uniform) >= 1:
                X_binned = np.digitize(X_col, thresholds_uniform).reshape(-1, 1).astype(float)
                clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                y_pred = cross_val_predict(clf, X_binned, y, cv=skf)
                score = f1_score(y, y_pred, zero_division=0)
                results.append({
                    'method': f'等距{n_bins}箱',
                    'thresholds': thresholds_uniform.tolist(),
                    'n_bins': len(thresholds_uniform) + 1,
                    'f1': score
                })
        except Exception:
            pass

    # ---- 方案 3：密集搜索单阈值（二分箱）----
    thresholds_candidates = np.percentile(X_col, np.arange(5, 96, 2.5))
    thresholds_candidates = np.unique(thresholds_candidates)

    for t in thresholds_candidates:
        X_binned = (X_col >= t).astype(float).reshape(-1, 1)
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_binned, y, cv=skf)
        score = f1_score(y, y_pred, zero_division=0)
        results.append({
            'method': f'单阈值',
            'thresholds': [round(t, 4)],
            'n_bins': 2,
            'f1': score
        })

    # ---- 方案 4：密集搜索双阈值（三分箱）----
    coarse_points = np.percentile(X_col, np.arange(10, 91, 5))
    coarse_points = np.unique(coarse_points)

    for i in range(len(coarse_points)):
        for j in range(i + 1, len(coarse_points)):
            t1, t2 = coarse_points[i], coarse_points[j]
            X_binned = np.digitize(X_col, [t1, t2]).reshape(-1, 1).astype(float)
            clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(clf, X_binned, y, cv=skf)
            score = f1_score(y, y_pred, zero_division=0)
            results.append({
                'method': f'双阈值',
                'thresholds': [round(t1, 4), round(t2, 4)],
                'n_bins': 3,
                'f1': score
            })

    # 排序
    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def search_all_features(X, y, feature_names, save_dir):
    """对每个特征搜索最优分箱方案"""
    print("\n" + "=" * 60)
    print("  🔍 各特征最优分箱搜索")
    print("=" * 60)

    all_results = {}

    for i, name in enumerate(feature_names):
        print(f"\n{'─' * 50}")
        print(f"  特征 [{name}]")
        print(f"{'─' * 50}")

        results = find_best_bins_single_feature(X[:, i], y)
        all_results[name] = results

        # 打印 Top 5
        print(f"\n    Top 5 分箱方案：")
        print(f"    {'排名':>4s} | {'方法':>8s} | {'箱数':>4s} | {'阈值':<30s} | {'F1':>6s}")
        print(f"    {'─'*4} | {'─'*8} | {'─'*4} | {'─'*30} | {'─'*6}")
        for rank, r in enumerate(results[:5], 1):
            thresh_str = ', '.join([f'{t:.3f}' for t in r['thresholds']])
            print(f"    {rank:>4d} | {r['method']:>8s} | {r['n_bins']:>4d} | "
                  f"{thresh_str:<30s} | {r['f1']:.4f}")

        # 最优
        best = results[0]
        print(f"\n    ⭐ 推荐：{best['method']} | 阈值={best['thresholds']} | F1={best['f1']:.4f}")

    return all_results


# ============================================================
#  4. 特征组合分箱搜索
# ============================================================
def search_feature_combinations(X, y, feature_names, all_results, save_dir):
    """
    用各特征的 Top 分箱方案组合起来，搜索最优整体方案
    """
    print("\n" + "=" * 60)
    print("  🔗 特征组合分箱搜索")
    print("=" * 60)

    # 每个特征取 Top 3 分箱方案 + 原始值
    per_feature_options = []
    for i, name in enumerate(feature_names):
        options = []

        # 选项 0：原始值
        options.append({
            'desc': f'{name}_原始',
            'data': X[:, i].reshape(-1, 1)
        })

        # 选项 1-3：Top 3 分箱
        for r in all_results[name][:3]:
            binned = np.digitize(X[:, i], r['thresholds']).astype(float).reshape(-1, 1)
            options.append({
                'desc': f"{name}_{r['method']}_{r['thresholds']}",
                'data': binned
            })

        per_feature_options.append(options)

    # 穷举组合（每个特征选一种方案）
    n_features = len(feature_names)
    n_options = [len(opts) for opts in per_feature_options]
    total_combos = 1
    for n in n_options:
        total_combos *= n

    print(f"\n    每个特征 {n_options} 种方案，共 {total_combos} 种组合")

    # 如果组合太多，随机采样
    max_combos = 2000
    if total_combos > max_combos:
        print(f"    组合数过多，随机采样 {max_combos} 种")

    best_combo_score = 0
    best_combo_desc = None
    best_combo_data = None
    searched = 0

    # 生成所有索引组合
    index_ranges = [range(len(opts)) for opts in per_feature_options]

    if total_combos <= max_combos:
        combos_iter = itertools.product(*index_ranges)
    else:
        # 随机采样
        rng = np.random.RandomState(42)
        combos_iter = [
            tuple(rng.randint(0, n) for n in n_options)
            for _ in range(max_combos)
        ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for combo_indices in combos_iter:
        searched += 1
        # 拼接各特征的选定方案
        parts = []
        descs = []
        for feat_idx, opt_idx in enumerate(combo_indices):
            opt = per_feature_options[feat_idx][opt_idx]
            parts.append(opt['data'])
            descs.append(opt['desc'])

        X_combo = np.hstack(parts)

        clf = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=42)
        y_pred = cross_val_predict(clf, X_combo, y, cv=skf)
        score = f1_score(y, y_pred, zero_division=0)

        if score > best_combo_score:
            best_combo_score = score
            best_combo_desc = descs
            best_combo_data = X_combo

        if searched % 500 == 0:
            print(f"    已搜索 {searched} 组，当前最优 F1={best_combo_score:.4f}")

    print(f"\n    总共搜索 {searched} 种组合")
    print(f"\n[✓] 最优组合 F1 = {best_combo_score:.4f}")
    print(f"    方案：")
    for d in best_combo_desc:
        print(f"      • {d}")

    return best_combo_score, best_combo_desc


# ============================================================
#  5. 生成推荐配置
# ============================================================
def generate_config_recommendation(all_results, feature_names, save_dir):
    """把搜索结果整理成可以直接粘贴到 config.py 的格式"""
    print("\n" + "=" * 60)
    print("  📋 推荐配置（可直接粘贴到 config.py）")
    print("=" * 60)

    lines = []
    lines.append("# ============================================================")
    lines.append("#  bin_optimizer.py 自动推荐的分箱阈值")
    lines.append("#  Auto-recommended binning thresholds")
    lines.append("# ============================================================\n")

    feature_to_config = {
        'SGR': 'SGR_BINS',
        '碳酸盐岩': 'ITE_BINS',
        '倾角中值': 'ANGLE_BINS',
        '断距中值': 'FAULT_BINS',
        '延伸长度(km)': 'LENGTH_BINS',
        '微裂缝': 'MICRO_BINS',
    }

    for name in feature_names:
        results = all_results.get(name, [])
        if not results:
            continue

        best = results[0]
        config_name = feature_to_config.get(name, f'{name}_BINS')

        thresholds = best['thresholds']
        bins_list = ['-float("inf")'] + [str(round(t, 4)) for t in thresholds] + ['float("inf")']
        bins_str = ', '.join(bins_list)

        lines.append(f"# {name}: 最优方案={best['method']}, F1={best['f1']:.4f}")
        lines.append(f"{config_name} = [{bins_str}]")
        lines.append("")

    config_text = '\n'.join(lines)
    print(f"\n{config_text}")

    # 保存到文件
    rec_path = os.path.join(save_dir, 'recommended_config.txt')
    with open(rec_path, 'w', encoding='utf-8') as f:
        f.write(config_text)
    print(f"\n[✓] 推荐配置已保存：{rec_path}")

    return config_text


# ============================================================
#  6. 可视化最优分箱效果
# ============================================================
def plot_best_binning(X, y, feature_names, all_results, save_dir):
    """可视化每个特征最优分箱后的类别分布"""
    n = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        results = all_results.get(name, [])
        if not results:
            continue

        best = results[0]
        thresholds = best['thresholds']

        vals = X[:, i]
        binned = np.digitize(vals, thresholds)

        # 每个箱的类别分布
        unique_bins = np.unique(binned)
        n_bins = len(unique_bins)

        bin_labels = []
        class0_counts = []
        class1_counts = []

        for b in unique_bins:
            mask = binned == b
            c0 = (y[mask] == 0).sum()
            c1 = (y[mask] == 1).sum()
            class0_counts.append(c0)
            class1_counts.append(c1)

            # 构造标签
            if b == 0:
                bin_labels.append(f'<{thresholds[0]:.2f}')
            elif b == len(thresholds):
                bin_labels.append(f'≥{thresholds[-1]:.2f}')
            else:
                bin_labels.append(f'{thresholds[b-1]:.2f}~{thresholds[b]:.2f}')

        x_pos = np.arange(n_bins)
        width = 0.35
        ax.bar(x_pos - width/2, class0_counts, width, label='封闭(0)', color='steelblue')
        ax.bar(x_pos + width/2, class1_counts, width, label='开启(1)', color='darkorange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=15)
        ax.set_title(f'{name}\n{best["method"]} F1={best["f1"]:.3f}', fontsize=10)
        ax.legend(fontsize=7)

        # 在柱子上标数字
        for x, c0, c1 in zip(x_pos, class0_counts, class1_counts):
            ax.text(x - width/2, c0 + 0.5, str(c0), ha='center', fontsize=8)
            ax.text(x + width/2, c1 + 0.5, str(c1), ha='center', fontsize=8)

    plt.suptitle('各特征最优分箱后的类别分布', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_binning.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
#  7. 原始特征 vs 分箱特征 对比
# ============================================================
def compare_raw_vs_binned(X, y, feature_names, all_results, save_dir):
    """对比：原始特征直接建树 vs 用最优分箱建树"""
    print("\n" + "=" * 60)
    print("  📊 原始 vs 分箱 对比")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loo = LeaveOneOut()

    comparison = []

    for i, name in enumerate(feature_names):
        # 原始
        X_raw_single = X[:, i].reshape(-1, 1)
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        y_pred_raw = cross_val_predict(clf, X_raw_single, y, cv=skf)
        f1_raw = f1_score(y, y_pred_raw, zero_division=0)

        # 分箱
        results = all_results.get(name, [])
        if results:
            best = results[0]
            X_binned = np.digitize(X[:, i], best['thresholds']).astype(float).reshape(-1, 1)
            clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
            y_pred_bin = cross_val_predict(clf, X_binned, y, cv=skf)
            f1_bin = f1_score(y, y_pred_bin, zero_division=0)
        else:
            f1_bin = 0

        diff = f1_bin - f1_raw
        marker = '⬆️' if diff > 0.01 else ('⬇️' if diff < -0.01 else '➡️')

        comparison.append({
            'name': name, 'f1_raw': f1_raw, 'f1_binned': f1_bin, 'diff': diff
        })

        print(f"    {name:16s} | 原始 F1={f1_raw:.4f} | 分箱 F1={f1_bin:.4f} | "
              f"差异={diff:+.4f} {marker}")

    # 全部特征一起对比
    print(f"\n    --- 全部特征一起 ---")

    # 原始
    clf = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=42)
    y_pred_raw = cross_val_predict(clf, X, y, cv=loo)
    f1_all_raw = f1_score(y, y_pred_raw, zero_division=0)
    acc_all_raw = accuracy_score(y, y_pred_raw)

    # 分箱
    X_all_binned_parts = []
    for i, name in enumerate(feature_names):
        results = all_results.get(name, [])
        if results and results[0]['f1'] > 0:
            binned = np.digitize(X[:, i], results[0]['thresholds']).astype(float)
        else:
            binned = X[:, i].copy()
        X_all_binned_parts.append(binned)
    X_all_binned = np.column_stack(X_all_binned_parts)

    clf = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=42)
    y_pred_bin = cross_val_predict(clf, X_all_binned, y, cv=loo)
    f1_all_bin = f1_score(y, y_pred_bin, zero_division=0)
    acc_all_bin = accuracy_score(y, y_pred_bin)

    # 混合：原始 + 分箱
    X_mixed = np.hstack([X, X_all_binned])
    clf = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=42)
    y_pred_mix = cross_val_predict(clf, X_mixed, y, cv=loo)
    f1_mixed = f1_score(y, y_pred_mix, zero_division=0)
    acc_mixed = accuracy_score(y, y_pred_mix)

    print(f"    全部原始       | LOOCV Acc={acc_all_raw:.4f}, F1={f1_all_raw:.4f}")
    print(f"    全部分箱       | LOOCV Acc={acc_all_bin:.4f}, F1={f1_all_bin:.4f}")
    print(f"    原始+分箱混合  | LOOCV Acc={acc_mixed:.4f}, F1={f1_mixed:.4f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c['name'] for c in comparison]
    raws = [c['f1_raw'] for c in comparison]
    bins = [c['f1_binned'] for c in comparison]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, raws, w, label='原始', color='steelblue', edgecolor='black')
    ax.bar(x + w/2, bins, w, label='分箱', color='darkorange', edgecolor='black')
    for xi, r, b in zip(x, raws, bins):
        ax.text(xi - w/2, r + 0.005, f'{r:.3f}', ha='center', fontsize=8)
        ax.text(xi + w/2, b + 0.005, f'{b:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('F1', fontsize=12)
    ax.set_title('单特征：原始 vs 最优分箱', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'raw_vs_binned.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
#  主函数
# ============================================================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  🔬 分箱阈值预分析工具")
    print("=" * 60)

    # 1. 加载数据
    X, y, df = load_data()

    # 2. 特征分布分析
    analyze_feature_distribution(X, y, BASE_FEATURES, OUTPUT_DIR)

    # 3. 每个特征搜索最优分箱
    all_results = search_all_features(X, y, BASE_FEATURES, OUTPUT_DIR)

    # 4. 可视化最优分箱
    plot_best_binning(X, y, BASE_FEATURES, all_results, OUTPUT_DIR)

    # 5. 原始 vs 分箱对比
    compare_raw_vs_binned(X, y, BASE_FEATURES, all_results, OUTPUT_DIR)

    # 6. 特征组合搜索
    best_combo_score, best_combo_desc = \
        search_feature_combinations(X, y, BASE_FEATURES, all_results, OUTPUT_DIR)

    # 7. 生成推荐配置
    config_text = generate_config_recommendation(all_results, BASE_FEATURES, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ✅ 分析完成！")
    print("=" * 60)
    print(f"\n  输出目录：{OUTPUT_DIR}/")
    print(f"  请将 {OUTPUT_DIR}/recommended_config.txt 中的阈值粘贴到 config.py")


if __name__ == '__main__':
    main()