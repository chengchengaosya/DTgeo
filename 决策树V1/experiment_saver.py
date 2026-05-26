"""
experiment_saver.py  v6.0
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


def create_experiment_dir():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    d = os.path.join('test', ts)
    os.makedirs(d, exist_ok=True)
    print(f"\n[✓] 实验目录：{d}")
    return d


def save_split_data(X_train, X_test, y_train, y_test, feature_names, save_dir):
    for data, labels, name in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
        df = pd.DataFrame(data, columns=feature_names)
        df['断层启闭性赋值'] = labels
        df.to_csv(os.path.join(save_dir, f'{name}_data.csv'), index=False, encoding='utf-8-sig')
    print(f"[✓] 数据已保存")


def save_predictions(X_test, y_test, y_pred, y_prob, feature_names, save_dir):
    df = pd.DataFrame(X_test, columns=feature_names)
    df['真实值'] = y_test
    df['预测值'] = y_pred
    df['预测概率'] = np.round(y_prob, 4)
    df['是否正确'] = (y_test == y_pred).astype(int)
    df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')


def _serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(i) for i in obj]
    return obj


def save_metrics(test_metrics, loocv_metrics, all_params, cv_scores, report,
                 roc_auc, ap, feature_names, save_dir):

    # JSON
    result = _serializable({
        'params': all_params,
        'test_metrics': {k: round(v, 4) for k, v in test_metrics.items()},
        'loocv_metrics': {k: round(v, 4) for k, v in loocv_metrics.items()},
        'roc_auc': round(roc_auc, 4),
        'ap': round(ap, 4),
        'cv': {m: {'mean': round(float(s.mean()), 4), 'std': round(float(s.std()), 4)}
               for m, s in cv_scores.items()},
        'features': list(feature_names),
    })
    with open(os.path.join(save_dir, 'experiment_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # TXT 双语报告
    sep = "=" * 65
    txt_path = os.path.join(save_dir, 'experiment_report.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{sep}\n")
        f.write("  断层启闭性分类 — 实验报告 v6.0\n")
        f.write("  Fault Sealing Classification — Report v6.0\n")
        f.write(f"{sep}\n")
        f.write(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  最终模型: {all_params.get('best_model', 'N/A')}\n\n")

        # 1. 各模型调参结果
        f.write(f"{sep}\n  1. 各模型调参结果 / Model Tuning Results\n{sep}\n\n")
        for name, info in all_params.items():
            if isinstance(info, dict) and 'cv_f1' in info:
                f.write(f"    {name}:\n")
                f.write(f"      CV F1: {info['cv_f1']}\n")
                if info.get('params'):
                    f.write(f"      参数: {info['params']}\n")
                f.write("\n")

        # 特征子集
        f.write(f"    最优特征子集 / Best Feature Subset:\n")
        f.write(f"      LOOCV F1: {all_params.get('feature_subset_loocv_f1', 'N/A')}\n")
        feats = all_params.get('best_features', feature_names)
        for i, name in enumerate(feats, 1):
            f.write(f"      {i}. {name}\n")

        # 2. LOOCV 评估（核心指标）
        f.write(f"\n{sep}\n  2. LOOCV 留一法评估（最可靠）\n")
        f.write(f"     LOOCV Evaluation (Most Reliable for Small Samples)\n{sep}\n\n")
        metric_desc = {
            'accuracy':          ('准确率',     'Accuracy'),
            'balanced_accuracy': ('平衡准确率', 'Balanced Accuracy'),
            'precision':         ('精确率',     'Precision'),
            'recall':            ('召回率',     'Recall'),
            'f1_score':          ('F1分数',     'F1-Score'),
            'mcc':               ('MCC',        'MCC'),
            'kappa':             ('Kappa',      "Cohen's Kappa"),
        }
        for key, val in loocv_metrics.items():
            cn, en = metric_desc.get(key, (key, key))
            f.write(f"    {cn} / {en}: {val:.4f}\n")

        # 3. 测试集评估
        f.write(f"\n{sep}\n  3. 测试集评估 / Test Set Metrics\n{sep}\n\n")
        metric_desc_full = {
            'accuracy':           ('准确率',     'Accuracy'),
            'balanced_accuracy':  ('平衡准确率', 'Balanced Accuracy'),
            'precision':          ('精确率',     'Precision'),
            'recall_sensitivity': ('召回率',     'Recall'),
            'specificity':        ('特异度',     'Specificity'),
            'npv':                ('NPV',        'NPV'),
            'f1_score':           ('F1',         'F1-Score'),
            'mcc':                ('MCC',        'MCC'),
            'kappa':              ('Kappa',      "Cohen's Kappa"),
            'log_loss':           ('Log Loss',   'Log Loss'),
            'brier_score':        ('Brier',      'Brier Score'),
        }
        for key, val in test_metrics.items():
            cn, en = metric_desc_full.get(key, (key, key))
            f.write(f"    {cn} / {en}: {val:.4f}\n")
        f.write(f"    ROC AUC: {roc_auc:.4f}\n")
        f.write(f"    AP: {ap:.4f}\n")

        # 4. 分类报告
        f.write(f"\n{sep}\n  4. 分类报告 / Classification Report\n{sep}\n\n")
        f.write(report + "\n")

        # 5. K折交叉验证
        f.write(f"{sep}\n  5. K折交叉验证 / K-Fold CV\n{sep}\n\n")
        for key, scores in cv_scores.items():
            f.write(f"    {key}: {scores.mean():.4f} ± {scores.std():.4f}\n")
            f.write(f"      各折: {', '.join([f'{s:.4f}' for s in scores])}\n\n")

        # 6. 解读
        f.write(f"{sep}\n  6. 指标解读 / Guide\n{sep}\n\n")
        for cond, desc in [
            ("Accuracy > 0.7",  "可接受 / Acceptable"),
            ("F1 > 0.7",        "均衡 / Balanced"),
            ("AUC > 0.8",       "强 / Strong"),
            ("MCC > 0.3",       "优于随机 / Better than random"),
            ("Kappa > 0.6",     "一致性好 / Substantial"),
        ]:
            f.write(f"    {cond:20s} → {desc}\n")

        f.write(f"\n{sep}\n  报告结束 / End\n{sep}\n")

    print(f"[✓] 报告已保存：{txt_path}")