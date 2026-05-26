"""
断层启闭性决策树分类 — 简洁版
只使用原始数值特征，不做特征工程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import json
import warnings
from datetime import datetime
from matplotlib.font_manager import FontProperties

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  配置
# ============================================================
DATA_PATH = '决策树—断层封闭性数据集.xlsx'
FEATURES = ['断距中值', '倾角中值', '延伸长度(km)', 'SGR', '碳酸盐岩', '微裂缝']
TARGET = '断层启闭性赋值'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_chinese_font():
    for name in ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong',
                 'KaiTi', 'Arial Unicode MS', 'Noto Sans CJK SC']:
        try:
            fp = FontProperties(family=name)
            if fp.get_name():
                return fp
        except:
            continue
    return FontProperties()


# ============================================================
#  1. 数据加载与校验
# ============================================================
def load_and_validate():
    print("=" * 50)
    print("  1. 数据加载与校验")
    print("=" * 50)

    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    print(f"[✓] 读取数据：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 校验列
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}")

    # 校验数值合法性
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[⚠] 删除 {dropped} 行无效数据")

    # 校验目标列
    if not set(df[TARGET].unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("目标列包含非 0/1 值")

    for col in FEATURES:
        df[col] = df[col].astype(np.float64)

    X = df[FEATURES].values
    y = df[TARGET].astype(int).values

    counts = np.bincount(y)
    print(f"[✓] 有效数据 {len(y)} 条 | 封闭(0): {counts[0]} | 开启(1): {counts[1]}")
    print("=" * 50)
    return X, y


# ============================================================
#  2. 划分数据集
# ============================================================
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[✓] 训练集 {len(y_train)} 条 | 测试集 {len(y_test)} 条")
    return X_train, X_test, y_train, y_test


# ============================================================
#  3. 网格搜索调参
# ============================================================
def tune(X_train, y_train):
    print("\n" + "=" * 50)
    print("  2. 网格搜索调参")
    print("=" * 50)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 5, 8, 10],
        'class_weight': [None, 'balanced'],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1',
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print(f"\n[✓] 最佳 CV F1：{grid.best_score_:.4f}")
    print(f"[✓] 最佳参数：")
    for k, v in grid.best_params_.items():
        print(f"    {k}: {v}")
    print(f"    树深度：{model.get_depth()}")
    print(f"    叶节点：{model.get_n_leaves()}")
    print("=" * 50)

    return model, grid.best_params_


# ============================================================
#  4. 模型评估
# ============================================================
def evaluate(model, X_test, y_test, save_dir):
    print("\n" + "=" * 50)
    print("  3. 测试集评估")
    print("=" * 50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n    Accuracy       ：{acc:.4f}")
    print(f"    Balanced Acc   ：{bal_acc:.4f}")
    print(f"    Precision      ：{prec:.4f}")
    print(f"    Recall         ：{rec:.4f}")
    print(f"    Specificity    ：{spec:.4f}")
    print(f"    F1-Score       ：{f1:.4f}")
    print(f"    MCC            ：{mcc:.4f}")
    print(f"    Kappa          ：{kappa:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=['封闭(0)', '开启(1)'],
        zero_division=0
    )
    print(f"\n{report}")

    metrics = {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'precision': prec, 'recall': rec,
        'specificity': spec, 'f1_score': f1,
        'mcc': mcc, 'kappa': kappa
    }
    print("=" * 50)
    return y_pred, y_prob, metrics, report


# ============================================================
#  5. 交叉验证
# ============================================================
def cross_validate(model, X, y):
    print("\n" + "=" * 50)
    print("  4. 10 折交叉验证")
    print("=" * 50)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for s in ['accuracy', 'f1', 'precision', 'recall']:
        scores = cross_val_score(model, X, y, cv=skf, scoring=s)
        results[s] = scores
        print(f"    {s:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    print("=" * 50)
    return results


# ============================================================
#  6. 可视化
# ============================================================
def plot_all(model, X, y, X_test, y_test, y_pred, y_prob, cv_scores, save_dir):
    print("\n[...] 生成可视化图表...")

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['封闭(0)', '开启(1)'],
                yticklabels=['封闭(0)', '开启(1)'],
                annot_kws={"size": 16})
    plt.xlabel('预测值', fontsize=13)
    plt.ylabel('真实值', fontsize=13)
    plt.title('混淆矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ---- ROC 曲线 ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel('TPR', fontsize=13)
    plt.title('ROC 曲线', fontsize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ---- PR 曲线 ----
    p_vals, r_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(r_vals, p_vals, 'steelblue', lw=2, label=f'AP = {ap:.4f}')
    plt.fill_between(r_vals, p_vals, alpha=0.1, color='steelblue')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('PR 曲线', fontsize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ---- 交叉验证柱状图 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, name) in zip(axes.flatten(),
            [('accuracy', '准确率'), ('f1', 'F1'), ('precision', '精确率'), ('recall', '召回率')]):
        scores = cv_scores[key]
        folds = [f'{i + 1}' for i in range(len(scores))]
        colors = ['#4CAF50' if s >= scores.mean() else '#FF7043' for s in scores]
        bars = ax.bar(folds, scores, color=colors, edgecolor='black')
        ax.axhline(y=scores.mean(), color='navy', ls='--', lw=1.5,
                   label=f'均值: {scores.mean():.3f}')
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(name)
        ax.legend(fontsize=9)
    plt.suptitle('10 折交叉验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_scores.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ---- 特征重要性 ----
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    names = [FEATURES[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(names)), vals, color='steelblue', edgecolor='black')
    plt.xticks(range(len(names)), names, fontsize=11)
    plt.ylabel('重要性', fontsize=13)
    plt.title('特征重要性', fontsize=15)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ---- 决策树结构图 ----
    cfp = get_chinese_font()
    depth = model.get_depth()
    n_leaves = model.get_n_leaves()
    fw = max(20, n_leaves * 3)
    fh = max(10, depth * 3)

    fig, ax = plt.subplots(figsize=(fw, fh))
    plot_tree(model, feature_names=FEATURES, class_names=['封闭', '开启'],
              filled=True, rounded=True, impurity=True,
              precision=3, ax=ax, fontsize=10)
    for t in ax.texts:
        t.set_fontproperties(cfp)
    ax.set_title('断层启闭性决策树', fontsize=20, fontweight='bold', fontproperties=cfp)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decision_tree.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return roc_auc, ap


# ============================================================
#  7. 保存结果
# ============================================================
def save_results(model, X_train, X_test, y_train, y_test, y_pred, y_prob,
                 best_params, metrics, cv_scores, report, roc_auc, ap, save_dir):
    print("\n[...] 保存实验结果...")

    # 训练集 / 测试集
    pd.DataFrame(X_train, columns=FEATURES).assign(**{TARGET: y_train}).to_csv(
        os.path.join(save_dir, 'train_data.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(X_test, columns=FEATURES).assign(**{TARGET: y_test}).to_csv(
        os.path.join(save_dir, 'test_data.csv'), index=False, encoding='utf-8-sig')

    # 预测结果
    df_pred = pd.DataFrame(X_test, columns=FEATURES)
    df_pred['真实值'] = y_test
    df_pred['预测值'] = y_pred
    df_pred['预测概率'] = np.round(y_prob, 4)
    df_pred['是否正确'] = (y_test == y_pred).astype(int)
    df_pred.to_csv(os.path.join(save_dir, 'predictions.csv'),
                   index=False, encoding='utf-8-sig')

    # 决策规则
    rules = export_text(model, feature_names=FEATURES, decimals=3)
    with open(os.path.join(save_dir, 'tree_rules.txt'), 'w', encoding='utf-8') as f:
        f.write(rules)

    # JSON
    result = {
        'best_params': {k: str(v) for k, v in best_params.items()},
        'tree_depth': model.get_depth(),
        'tree_leaves': model.get_n_leaves(),
        'test_metrics': {k: round(v, 4) for k, v in metrics.items()},
        'roc_auc': round(roc_auc, 4),
        'ap': round(ap, 4),
        'cv': {m: {'mean': round(float(s.mean()), 4), 'std': round(float(s.std()), 4)}
               for m, s in cv_scores.items()},
    }
    with open(os.path.join(save_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 双语 TXT 报告
    sep = "=" * 60
    with open(os.path.join(save_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{sep}\n")
        f.write("  断层启闭性决策树 — 实验报告\n")
        f.write("  Fault Sealing Decision Tree — Report\n")
        f.write(f"{sep}\n")
        f.write(f"  时间/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{sep}\n  1. 最佳参数 / Best Parameters\n{sep}\n\n")
        for k, v in best_params.items():
            f.write(f"    {k}: {v}\n")
        f.write(f"    树深度/Depth: {model.get_depth()}\n")
        f.write(f"    叶节点/Leaves: {model.get_n_leaves()}\n")
        f.write(f"    使用特征/Features: {FEATURES}\n")

        f.write(f"\n{sep}\n  2. 测试集指标 / Test Metrics\n{sep}\n\n")
        desc = {
            'accuracy': ('准确率', 'Accuracy', '正确预测比例', 'Correct predictions ratio'),
            'balanced_accuracy': ('平衡准确率', 'Balanced Acc', '各类召回均值', 'Mean per-class recall'),
            'precision': ('精确率', 'Precision', '预测正中真正', 'TP / predicted positives'),
            'recall': ('召回率', 'Recall', '真正中被找到', 'TP / actual positives'),
            'specificity': ('特异度', 'Specificity', '真负中被找到', 'TN / actual negatives'),
            'f1_score': ('F1', 'F1-Score', '精确率召回率调和', 'Harmonic mean of P and R'),
            'mcc': ('MCC', 'MCC', '综合相关系数[-1,1]', 'Correlation coefficient [-1,1]'),
            'kappa': ('Kappa', "Cohen's Kappa", '扣除随机一致性', 'Chance-corrected agreement'),
        }
        for k, v in metrics.items():
            if k in desc:
                cn, en, cd, ed = desc[k]
                f.write(f"    {cn} / {en}: {v:.4f}\n")
                f.write(f"      ↳ {cd} / {ed}\n\n")
        f.write(f"    ROC AUC: {roc_auc:.4f}\n")
        f.write(f"      ↳ ROC曲线下面积 / Area under ROC\n\n")
        f.write(f"    AP: {ap:.4f}\n")
        f.write(f"      ↳ PR曲线下面积 / Area under PR\n\n")

        f.write(f"{sep}\n  3. 分类报告 / Classification Report\n{sep}\n\n")
        f.write(report + "\n")

        f.write(f"{sep}\n  4. 10折交叉验证 / 10-Fold CV\n{sep}\n\n")
        cv_desc = {'accuracy': '准确率/Acc', 'f1': 'F1', 'precision': '精确率/Prec', 'recall': '召回率/Rec'}
        for k, scores in cv_scores.items():
            f.write(f"    {cv_desc.get(k, k)}: {scores.mean():.4f} ± {scores.std():.4f}\n")
            f.write(f"      各折: {', '.join(f'{s:.4f}' for s in scores)}\n\n")

        f.write(f"{sep}\n  5. 决策规则 / Decision Rules\n{sep}\n\n")
        f.write(rules + "\n")

        f.write(f"{sep}\n  6. 指标解读 / Guide\n{sep}\n\n")
        for c, d in [("Accuracy > 0.7", "可接受/Acceptable"), ("F1 > 0.7", "均衡/Balanced"),
                     ("AUC > 0.8", "强/Strong"), ("MCC > 0.3", "优于随机/Better than random"),
                     ("Kappa > 0.6", "一致性好/Substantial")]:
            f.write(f"    {c:20s} → {d}\n")
        f.write(f"\n{sep}\n  报告结束 / End\n{sep}\n")

    print(f"[✓] 全部结果已保存至 {save_dir}/")


# ============================================================
#  主流程
# ============================================================
def main():
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('test', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[✓] 实验目录：{save_dir}\n")

    # 1. 加载数据
    X, y = load_and_validate()

    # 2. 划分
    X_train, X_test, y_train, y_test = split(X, y)

    # 3. 调参
    model, best_params = tune(X_train, y_train)

    # 4. 评估
    y_pred, y_prob, metrics, report = evaluate(model, X_test, y_test, save_dir)

    # 5. 交叉验证
    cv_scores = cross_validate(model, X, y)

    # 6. 可视化
    roc_auc, ap = plot_all(model, X, y, X_test, y_test, y_pred, y_prob, cv_scores, save_dir)

    # 7. 保存
    save_results(model, X_train, X_test, y_train, y_test, y_pred, y_prob,
                 best_params, metrics, cv_scores, report, roc_auc, ap, save_dir)

    print(f"\n🎉 完成！")


if __name__ == '__main__':
    main()