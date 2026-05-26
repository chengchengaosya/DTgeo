"""
model_builder.py
专注决策树：Optuna 深度调参 + Bagging 增强
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut,
    RepeatedStratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import accuracy_score, f1_score
import config

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("[✗] 请安装 optuna: pip install optuna")
    import sys; sys.exit(1)


def split_data(X, y):
    """多种子寻找最优划分"""
    best_seed, best_diff = 42, float('inf')
    ratio = y.mean()
    for seed in range(config.SPLIT_SEED_TRIALS):
        _, _, _, yt = train_test_split(X, y, test_size=config.TEST_SIZE,
                                       random_state=seed, stratify=y)
        diff = abs(yt.mean() - ratio)
        if diff < best_diff:
            best_diff = diff
            best_seed = seed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=best_seed, stratify=y
    )
    print(f"\n[✓] 数据划分（种子={best_seed}）：")
    print(f"    训练 {X_train.shape[0]} | 测试 {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def _parse_class_weight(cw_str):
    """解析类别权重字符串"""
    mapping = {
        'None': None,
        'balanced': 'balanced',
        '1_1.5': {0: 1, 1: 1.5},
        '1_2': {0: 1, 1: 2},
        '1.5_1': {0: 1.5, 1: 1},
        '1_3': {0: 1, 1: 3},
    }
    return mapping.get(cw_str, None)


def tune_decision_tree(X_train, y_train):
    """Optuna 贝叶斯调参决策树"""
    print("\n" + "=" * 55)
    print("          🔧 决策树 Optuna 调参")
    print("=" * 55)

    cv = RepeatedStratifiedKFold(
        n_splits=config.TUNE_CV_FOLDS,
        n_repeats=config.TUNE_CV_REPEATS,
        random_state=42
    )

    def objective(trial):
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_categorical('max_depth', config.MAX_DEPTH_CANDIDATES),
            'min_samples_split': trial.suggest_int(
                'min_samples_split', *config.MIN_SAMPLES_SPLIT_RANGE),
            'min_samples_leaf': trial.suggest_int(
                'min_samples_leaf', *config.MIN_SAMPLES_LEAF_RANGE),
            'ccp_alpha': trial.suggest_float(
                'ccp_alpha', *config.CCP_ALPHA_RANGE, step=config.CCP_ALPHA_STEP),
        }
        cw_str = trial.suggest_categorical('class_weight_str', config.CLASS_WEIGHT_OPTIONS)
        cw = _parse_class_weight(cw_str)

        clf = DecisionTreeClassifier(**params, class_weight=cw, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)

    # 提取最佳参数
    bp = study.best_params.copy()
    cw_str = bp.pop('class_weight_str')
    bp['class_weight'] = _parse_class_weight(cw_str)

    # 训练最优决策树
    best_tree = DecisionTreeClassifier(**bp, random_state=42)
    best_tree.fit(X_train, y_train)

    print(f"\n[✓] 最佳 CV F1：{study.best_value:.4f}")
    print(f"    参数：")
    for k, v in bp.items():
        print(f"      {k}: {v}")
    print(f"    树深度：{best_tree.get_depth()}")
    print(f"    叶节点：{best_tree.get_n_leaves()}")

    return best_tree, bp, study.best_value


def build_bagging_tree(best_tree_params, X_train, y_train):
    """
    用调优后的决策树做 Bagging
    关键区别于随机森林：
    - 随机森林的每棵树用随机参数
    - 这里每棵树都用相同的最优参数，只是训练数据不同
    """
    if not config.USE_BAGGING:
        return None, 0

    print("\n[...] 构建 Bagging 决策树...")

    base_tree = DecisionTreeClassifier(**best_tree_params, random_state=42)

    bagging = BaggingClassifier(
        estimator=base_tree,
        n_estimators=config.BAGGING_N_ESTIMATORS,
        max_samples=config.BAGGING_MAX_SAMPLES,
        max_features=config.BAGGING_MAX_FEATURES,
        bootstrap=True,
        oob_score=True,     # 袋外评分
        random_state=42,
        n_jobs=-1
    )
    bagging.fit(X_train, y_train)

    oob = bagging.oob_score_
    print(f"    OOB 准确率：{oob:.4f}")

    # LOOCV 评估
    loo = LeaveOneOut()
    y_pred = cross_val_predict(bagging, X_train, y_train, cv=loo)
    bag_f1 = f1_score(y_train, y_pred, zero_division=0)
    bag_acc = accuracy_score(y_train, y_pred)
    print(f"    Bagging LOOCV: Acc={bag_acc:.4f}, F1={bag_f1:.4f}")

    return bagging, bag_f1


def build_and_tune(X_train, y_train, feature_names):
    """
    完整流程：
    1. Optuna 调参单棵决策树
    2. Bagging 增强
    3. 选最优
    """
    print("\n" + "=" * 55)
    print("          🔧 模型训练")
    print("=" * 55)

    # 1. 调参
    best_tree, best_params, tree_cv_f1 = tune_decision_tree(X_train, y_train)

    # 单棵树 LOOCV
    loo = LeaveOneOut()
    y_pred_tree = cross_val_predict(best_tree, X_train, y_train, cv=loo)
    tree_loocv_f1 = f1_score(y_train, y_pred_tree, zero_division=0)
    tree_loocv_acc = accuracy_score(y_train, y_pred_tree)
    print(f"\n    单棵决策树 LOOCV: Acc={tree_loocv_acc:.4f}, F1={tree_loocv_f1:.4f}")

    # 2. Bagging
    bagging_model, bag_f1 = build_bagging_tree(best_params, X_train, y_train)

    # 3. 选最优
    if bagging_model and bag_f1 > tree_loocv_f1:
        final_model = bagging_model
        use_bagging = True
        final_f1 = bag_f1
        print(f"\n[✓] 选用 Bagging 决策树（F1: {tree_loocv_f1:.4f} → {bag_f1:.4f}）")
    else:
        final_model = best_tree
        use_bagging = False
        final_f1 = tree_loocv_f1
        print(f"\n[✓] 选用单棵决策树（F1={tree_loocv_f1:.4f}）")

    all_params = {
        'tree_params': {k: str(v) if isinstance(v, dict) else v
                        for k, v in best_params.items()},
        'tree_cv_f1': round(tree_cv_f1, 4),
        'tree_loocv_f1': round(tree_loocv_f1, 4),
        'tree_loocv_acc': round(tree_loocv_acc, 4),
        'bagging_loocv_f1': round(bag_f1, 4) if bagging_model else None,
        'use_bagging': use_bagging,
    }

    print("=" * 55)
    return final_model, best_tree, best_params, all_params