"""
model_evaluator.py  v6.0
LOOCV 为主评估 + 测试集辅助验证 + 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss, brier_score_loss
)
from sklearn.model_selection import (
    cross_val_score, cross_val_predict,
    StratifiedKFold, LeaveOneOut, learning_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def evaluate_test_set(model, X_test, y_test):
    """测试集评估"""
    print("\n" + "=" * 55)
    print("          📊 测试集评估")
    print("=" * 55)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n    准确率     Accuracy    ：{acc:.4f}")
    print(f"    平衡准确率  Bal. Acc   ：{bal_acc:.4f}")
    print(f"    精确率     Precision   ：{prec:.4f}")
    print(f"    召回率     Recall      ：{rec:.4f}")
    print(f"    特异度     Specificity ：{specificity:.4f}")
    print(f"    F1                     ：{f1:.4f}")
    print(f"    MCC                    ：{mcc:.4f}")
    print(f"    Kappa                  ：{kappa:.4f}")
    print(f"    Log Loss               ：{logloss:.4f}")
    print(f"    Brier                  ：{brier:.4f}")

    target_names = ['封闭 (0)', '开启 (1)']
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(f"\n{report}")

    metrics = {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'precision': prec, 'recall_sensitivity': rec,
        'specificity': specificity, 'npv': npv,
        'f1_score': f1, 'mcc': mcc, 'kappa': kappa,
        'log_loss': logloss, 'brier_score': brier
    }
    return y_pred, y_prob, metrics, report


def loocv_evaluate_full(model, X, y):
    """
    LOOCV 留一法全面评估（小样本最可靠的评估方式）
    每次留 1 条做测试，其余全部训练，重复 N 次
    """
    print("\n" + "=" * 55)
    print(f"          🔁 LOOCV 留一法评估（{len(y)} 轮）")
    print("=" * 55)

    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    print(f"\n    准确率     Accuracy  ：{acc:.4f}")
    print(f"    平衡准确率  Bal. Acc ：{bal_acc:.4f}")
    print(f"    精确率     Precision ：{prec:.4f}")
    print(f"    召回率     Recall    ：{rec:.4f}")
    print(f"    F1                   ：{f1:.4f}")
    print(f"    MCC                  ：{mcc:.4f}")
    print(f"    Kappa                ：{kappa:.4f}")
    print("=" * 55)

    loocv_metrics = {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'precision': prec, 'recall': rec,
        'f1_score': f1, 'mcc': mcc, 'kappa': kappa
    }
    return y_pred, loocv_metrics


def plot_confusion_matrix(y_true, y_pred, title, save_dir, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['封闭 (0)', '开启 (1)'],
                yticklabels=['封闭 (0)', '开启 (1)'],
                annot_kws={"size": 16})
    plt.xlabel('预测值', fontsize=13)
    plt.ylabel('真实值', fontsize=13)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_test, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel('TPR', fontsize=13)
    plt.title('ROC 曲线', fontsize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return roc_auc


def plot_pr_curve(y_test, y_prob, save_dir):
    p, r, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(r, p, color='steelblue', lw=2, label=f'AP = {ap:.4f}')
    plt.fill_between(r, p, alpha=0.1, color='steelblue')
    plt.xlabel('召回率', fontsize=13)
    plt.ylabel('精确率', fontsize=13)
    plt.title('PR 曲线', fontsize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return ap


def cross_validate_kfold(model, X, y, cv=10):
    """K 折交叉验证"""
    print("\n" + "=" * 55)
    print(f"          🔁 {cv} 折交叉验证")
    print("=" * 55)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    for s in ['accuracy', 'f1', 'precision', 'recall']:
        scores = cross_val_score(model, X, y, cv=skf, scoring=s)
        results[s] = scores
        print(f"    {s:12s}: {scores.mean():.4f} ± {scores.std():.4f}")
    print("=" * 55)
    return results


def plot_cv_scores(cv_scores, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, name) in zip(axes.flatten(),
            [('accuracy','准确率'),('f1','F1'),('precision','精确率'),('recall','召回率')]):
        scores = cv_scores[key]
        folds = [f'{i+1}' for i in range(len(scores))]
        colors = ['#4CAF50' if s >= scores.mean() else '#FF7043' for s in scores]
        bars = ax.bar(folds, scores, color=colors, edgecolor='black')
        ax.axhline(y=scores.mean(), color='navy', ls='--', lw=1.5,
                   label=f'均值:{scores.mean():.3f}')
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.2f}', ha='center', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(name)
        ax.legend(fontsize=9)
    plt.suptitle('K折交叉验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_scores.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, save_dir):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]
    vals = importances[indices]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(len(names)), vals, color='steelblue', edgecolor='black')
    plt.xticks(range(len(names)), names, fontsize=10, rotation=30, ha='right')
    plt.ylabel('重要性', fontsize=13)
    plt.title('特征重要性', fontsize=15)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_learning_curve(model, X, y, save_dir):
    print("\n[...] 学习曲线...")
    train_sizes, train_s, test_s = learning_curve(
        model, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    plt.fill_between(train_sizes, train_s.mean(1)-train_s.std(1),
                     train_s.mean(1)+train_s.std(1), alpha=0.15, color='steelblue')
    plt.fill_between(train_sizes, test_s.mean(1)-test_s.std(1),
                     test_s.mean(1)+test_s.std(1), alpha=0.15, color='darkorange')
    plt.plot(train_sizes, train_s.mean(1), 'o-', color='steelblue', lw=2, label='训练集')
    plt.plot(train_sizes, test_s.mean(1), 'o-', color='darkorange', lw=2, label='验证集')
    plt.xlabel('样本数', fontsize=13)
    plt.ylabel('准确率', fontsize=13)
    plt.title('学习曲线', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(model_results, save_dir):
    """多模型对比柱状图"""
    names = list(model_results.keys())
    scores = [model_results[n]['score'] for n in names]

    plt.figure(figsize=(10, 5))
    colors = ['#2196F3' if s == max(scores) else '#90CAF9' for s in scores]
    bars = plt.bar(names, scores, color=colors, edgecolor='black')
    for bar, val in zip(bars, scores):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    plt.ylabel('CV F1 分数', fontsize=13)
    plt.title('多模型 Optuna 调参后 CV F1 对比', fontsize=15)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()