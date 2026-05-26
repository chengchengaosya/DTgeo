"""
model_builder.py  v6.0
核心策略：
1. 穷举特征子集，找最优组合
2. 允许深树（max_depth 最高不限）
3. LOOCV 留一法评估（小样本最优策略）
4. SVM/KNN 加入候选（小样本利器）
5. 软投票集成
"""

import numpy as np
import warnings
import itertools
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut,
    cross_val_score, cross_val_predict, RepeatedStratifiedKFold
)
from sklearn.metrics import accuracy_score, f1_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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


def split_data(X, y, test_size=0.2):
    """多种子选最优划分"""
    best_seed, best_diff = 42, float('inf')
    ratio = y.mean()
    for seed in range(100):
        _, _, _, yt = train_test_split(X, y, test_size=test_size,
                                       random_state=seed, stratify=y)
        diff = abs(yt.mean() - ratio)
        if diff < best_diff:
            best_diff = diff
            best_seed = seed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=best_seed, stratify=y
    )
    print(f"\n[✓] 数据划分（种子={best_seed}）：")
    print(f"    训练 {X_train.shape[0]} 条 | 测试 {X_test.shape[0]} 条")
    return X_train, X_test, y_train, y_test


def find_best_feature_subset(X, y, feature_names, max_combo_size=6):
    """
    穷举特征子集（2~max_combo_size），用 LOOCV 选最优组合
    这是小样本下最可靠的特征选择方法
    """
    print("\n" + "=" * 55)
    print("          🔍 穷举特征子集（LOOCV）")
    print("=" * 55)

    n_features = len(feature_names)
    loo = LeaveOneOut()

    best_score = 0
    best_indices = None
    best_size = 0
    total_combos = 0

    # 限制组合数量，优先搜索小特征子集
    for size in range(2, min(max_combo_size + 1, n_features + 1)):
        combos = list(itertools.combinations(range(n_features), size))
        total_combos += len(combos)

        # 如果组合数太多（>500），随机采样
        if len(combos) > 500:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(combos), 500, replace=False)
            combos = [combos[i] for i in sample_idx]

        for indices in combos:
            X_sub = X[:, indices]
            # 用简单决策树做 LOOCV（速度快）
            clf = DecisionTreeClassifier(
                max_depth=None, class_weight='balanced', random_state=42
            )
            y_pred = cross_val_predict(clf, X_sub, y, cv=loo)
            score = f1_score(y, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_indices = indices
                best_size = size

    selected_names = [feature_names[i] for i in best_indices]
    print(f"\n[✓] 搜索了约 {total_combos} 种组合")
    print(f"[✓] 最优子集（{best_size}个特征）LOOCV F1 = {best_score:.4f}")
    print(f"    特征：{selected_names}")

    return list(best_indices), selected_names, best_score


def loocv_evaluate(model, X, y):
    """留一法评估，返回准确率和 F1"""
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)
    return acc, f1


def optuna_tune_all_models(X_train, y_train, n_trials=150):
    """
    对多种模型分别用 Optuna 调参
    关键：决策树允许深树（max_depth 可达 20 或 None）
    """
    print("\n" + "=" * 55)
    print("          🔧 Optuna 多模型调参")
    print("=" * 55)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    results = {}

    # ==========================================
    #  1. 决策树（允许深树）
    # ==========================================
    print("\n[1/5] 决策树（允许深树）...")

    def dt_objective(trial):
        p = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_categorical('max_depth',
                                                    [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 50]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 25),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.03, step=0.001),
            'class_weight': trial.suggest_categorical('class_weight_str',
                                                       ['None', 'balanced']),
        }
        cw = None if p.pop('class_weight') == 'None' else 'balanced'
        clf = DecisionTreeClassifier(**p, class_weight=cw, random_state=42)
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

    study_dt = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_dt.optimize(dt_objective, n_trials=n_trials)
    bp = study_dt.best_params.copy()
    cw_str = bp.pop('class_weight_str')
    bp['class_weight'] = None if cw_str == 'None' else 'balanced'
    dt_model = DecisionTreeClassifier(**bp, random_state=42)
    dt_model.fit(X_train, y_train)
    results['决策树'] = {'model': dt_model, 'score': study_dt.best_value, 'params': bp}
    print(f"    最佳 F1: {study_dt.best_value:.4f} | depth={bp.get('max_depth')}")

    # ==========================================
    #  2. SVM（小样本利器）
    # ==========================================
    print("[2/5] SVM...")

    def svm_objective(trial):
        p = {
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('cw', ['None', 'balanced']),
        }
        cw = None if p.pop('class_weight') == 'None' else 'balanced'
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**p, class_weight=cw, probability=True, random_state=42))
        ])
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

    study_svm = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_svm.optimize(svm_objective, n_trials=n_trials)
    bp_svm = study_svm.best_params.copy()
    cw_svm = None if bp_svm.pop('cw') == 'None' else 'balanced'
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**bp_svm, class_weight=cw_svm, probability=True, random_state=42))
    ])
    svm_model.fit(X_train, y_train)
    results['SVM'] = {'model': svm_model, 'score': study_svm.best_value, 'params': bp_svm}
    print(f"    最佳 F1: {study_svm.best_value:.4f}")

    # ==========================================
    #  3. KNN（小样本友好）
    # ==========================================
    print("[3/5] KNN...")

    def knn_objective(trial):
        p = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 25, step=2),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        }
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**p))
        ])
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

    study_knn = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_knn.optimize(knn_objective, n_trials=80)
    bp_knn = study_knn.best_params.copy()
    knn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**bp_knn))
    ])
    knn_model.fit(X_train, y_train)
    results['KNN'] = {'model': knn_model, 'score': study_knn.best_value, 'params': bp_knn}
    print(f"    最佳 F1: {study_knn.best_value:.4f}")

    # ==========================================
    #  4. 随机森林
    # ==========================================
    print("[4/5] 随机森林...")

    def rf_objective(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth',
                                                    [3, 5, 7, 10, 15, 20, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('cw', ['None', 'balanced',
                                                              'balanced_subsample']),
        }
        cw = None if p.pop('class_weight') == 'None' else p.get('class_weight', 'balanced')
        # 修正：从 suggest 取出
        clf = RandomForestClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_split=p['min_samples_split'],
            min_samples_leaf=p['min_samples_leaf'],
            class_weight=cw, random_state=42, n_jobs=-1
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

    study_rf = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_rf.optimize(rf_objective, n_trials=n_trials)
    bp_rf = study_rf.best_params.copy()
    cw_rf_str = bp_rf.pop('cw')
    cw_rf = None if cw_rf_str == 'None' else cw_rf_str
    rf_model = RandomForestClassifier(
        n_estimators=bp_rf['n_estimators'],
        max_depth=bp_rf['max_depth'],
        min_samples_split=bp_rf['min_samples_split'],
        min_samples_leaf=bp_rf['min_samples_leaf'],
        class_weight=cw_rf, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    results['随机森林'] = {'model': rf_model, 'score': study_rf.best_value, 'params': bp_rf}
    print(f"    最佳 F1: {study_rf.best_value:.4f}")

    # ==========================================
    #  5. XGBoost / LightGBM
    # ==========================================
    if HAS_XGB:
        print("[5a/5] XGBoost...")
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        def xgb_objective(trial):
            p = {
                'n_estimators': trial.suggest_int('n_estimators', 30, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, spw * 2),
            }
            clf = XGBClassifier(**p, random_state=42, eval_metric='logloss', verbosity=0)
            return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

        study_xgb = optuna.create_study(direction='maximize',
                                         sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(xgb_objective, n_trials=n_trials)
        xgb_model = XGBClassifier(**study_xgb.best_params, random_state=42,
                                   eval_metric='logloss', verbosity=0)
        xgb_model.fit(X_train, y_train)
        results['XGBoost'] = {'model': xgb_model, 'score': study_xgb.best_value,
                               'params': study_xgb.best_params}
        print(f"    最佳 F1: {study_xgb.best_value:.4f}")

    if HAS_LGBM:
        print("[5b/5] LightGBM...")

        def lgbm_objective(trial):
            p = {
                'n_estimators': trial.suggest_int('n_estimators', 30, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
            }
            clf = LGBMClassifier(**p, random_state=42, verbosity=-1)
            return cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1').mean()

        study_lgbm = optuna.create_study(direction='maximize',
                                          sampler=optuna.samplers.TPESampler(seed=42))
        study_lgbm.optimize(lgbm_objective, n_trials=n_trials)
        lgbm_model = LGBMClassifier(**study_lgbm.best_params, random_state=42, verbosity=-1)
        lgbm_model.fit(X_train, y_train)
        results['LightGBM'] = {'model': lgbm_model, 'score': study_lgbm.best_value,
                                'params': study_lgbm.best_params}
        print(f"    最佳 F1: {study_lgbm.best_value:.4f}")

    return results


def build_voting_ensemble(model_results, X_train, y_train):
    """
    软投票集成：取 Top-3 模型做加权投票
    权重 = 各模型 CV F1 分数
    """
    print("\n[...] 构建软投票集成（Top-3 模型）...")

    # 按 F1 排序取 Top 3
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['score'], reverse=True)
    top3 = sorted_models[:3]

    estimators = []
    weights = []
    for name, info in top3:
        estimators.append((name, info['model']))
        weights.append(info['score'])
        print(f"    {name}: F1={info['score']:.4f}")

    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    voting.fit(X_train, y_train)

    # LOOCV 评估
    loo = LeaveOneOut()
    y_pred = cross_val_predict(voting, X_train, y_train, cv=loo)
    voting_f1 = f1_score(y_train, y_pred, zero_division=0)
    voting_acc = accuracy_score(y_train, y_pred)
    print(f"\n    软投票 LOOCV: Acc={voting_acc:.4f}, F1={voting_f1:.4f}")

    return voting, voting_acc, voting_f1


def build_and_tune(X_train, y_train, feature_names):
    """
    v6.0 完整流程：
    1. 穷举特征子集
    2. 在最优子集上 Optuna 调参多种模型
    3. 软投票集成 Top-3
    4. 选最终模型
    """
    print("\n" + "=" * 55)
    print("          🔧 模型训练 v6.0")
    print("=" * 55)

    # ---- 1. 穷举特征子集 ----
    best_feat_idx, best_feat_names, subset_score = \
        find_best_feature_subset(X_train, y_train, feature_names, max_combo_size=6)

    X_train_sel = X_train[:, best_feat_idx]

    # ---- 2. 多模型 Optuna 调参 ----
    model_results = optuna_tune_all_models(X_train_sel, y_train, n_trials=150)

    # ---- 3. 软投票集成 ----
    voting_model, voting_acc, voting_f1 = \
        build_voting_ensemble(model_results, X_train_sel, y_train)

    # ---- 4. 汇总选最优 ----
    print("\n" + "-" * 40)
    print("  各模型汇总：")
    all_candidates = {}
    for name, info in model_results.items():
        all_candidates[name] = info
        print(f"    {name:12s}: CV F1 = {info['score']:.4f}")
    all_candidates['软投票集成'] = {
        'model': voting_model, 'score': voting_f1, 'params': {}
    }
    print(f"    {'软投票集成':12s}: LOOCV F1 = {voting_f1:.4f}")

    best_name = max(all_candidates, key=lambda k: all_candidates[k]['score'])
    final_model = all_candidates[best_name]['model']
    final_score = all_candidates[best_name]['score']

    print(f"\n[✓] 最终选用：{best_name}（F1={final_score:.4f}）")

    # 提取决策树模型（用于可视化）
    dt_model = model_results['决策树']['model']

    # 汇总参数
    all_params = {
        name: {
            'cv_f1': round(info['score'], 4),
            'params': {k: str(v) if isinstance(v, dict) else v
                       for k, v in info.get('params', {}).items()}
        }
        for name, info in all_candidates.items()
    }
    all_params['best_model'] = best_name
    all_params['best_features'] = best_feat_names
    all_params['feature_subset_loocv_f1'] = round(subset_score, 4)

    print("=" * 55)

    return (final_model, dt_model, all_params, best_feat_idx,
            best_feat_names, model_results)