"""
experiment_saver.py
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import config


def create_experiment_dir():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    d = os.path.join(config.OUTPUT_DIR, ts)
    os.makedirs(d, exist_ok=True)
    print(f"\n[✓] 实验目录：{d}")
    return d


def save_split_data(X_train, X_test, y_train, y_test, feature_names, save_dir):
    for data, labels, name in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
        df = pd.DataFrame(data, columns=feature_names)
        df[config.TARGET_COL] = labels
        df.to_csv(os.path.join(save_dir, f'{name}_data.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"[✓] 数据已保存")


def save_predictions(X_test, y_test, y_pred, y_prob, feature_names, save_dir):
    df = pd.DataFrame(X_test, columns=feature_names)
    df['真实值'] = y_test
    df['预测值'] = y_pred
    df['预测概率'] = np.round(y_prob, 4)
    df['是否正确'] = (y_test == y_pred).astype(int)
    df.to_csv(os.path.join(save_dir, 'predictions.csv'),
              index=False, encoding='utf-8-sig')


def _ser(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {str(k): _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_ser(i) for i in obj]
    return obj


def save_metrics(test_metrics, loocv_metrics, all_params, cv_scores, report,
                 roc_auc, ap, feature_names, subset_f1, save_dir):

    # JSON
    result = _ser({
        'params': all_params,
        'feature_subset_loocv_f1': round(subset_f1, 4),
        'test_metrics': {k: round(v, 4) for k, v in test_metrics.items()},
        'loocv_metrics': {k: round(v, 4) for k, v in loocv_metrics.items()},
        'roc_auc': round(roc_auc, 4), 'ap': round(ap, 4),
        'cv': {m: {'mean': round(float(s.mean()), 4), 'std': round(float(s.std()), 4)}
               for m, s in cv_scores.items()},
        'features': list(feature_names),
    })
    with open(os.path.join(save_dir, 'experiment_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # TXT 双语报告
    sep = "=" * 65
    txt = os.path.join(save_dir, 'experiment_report.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write(f"{sep}\n")
        f.write("  断层启闭性决策树 — 实验报告 v7.0\n")
        f.write("  Fault Sealing Decision Tree — Report v7.0\n")
        f.write(f"{sep}\n")
        f.write(f"  时间/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  使用Bagging: {all_params.get('use_bagging', False)}\n\n")

        # 1. 特征与参数
        f.write(f"{sep}\n  1. 配置 / Configuration\n{sep}\n\n")
        f.write(f"    特征子集 LOOCV F1 / Feature Subset LOOCV F1: {subset_f1:.4f}\n")
        f.write(f"    选用特征 / Selected Features:\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"      {i}. {name}\n")
        f.write(f"\n    决策树参数 / Tree Params:\n")
        tp = all_params.get('tree_params', {})
        for k, v in tp.items():
            f.write(f"      {k}: {v}\n")
        f.write(f"\n    决策树 LOOCV / Tree LOOCV:\n")
        f.write(f"      Acc: {all_params.get('tree_loocv_acc', 'N/A')}\n")
        f.write(f"      F1:  {all_params.get('tree_loocv_f1', 'N/A')}\n")
        if all_params.get('bagging_loocv_f1'):
            f.write(f"    Bagging LOOCV F1: {all_params['bagging_loocv_f1']}\n")

        # 2. LOOCV
        f.write(f"\n{sep}\n  2. LOOCV 评估 / LOOCV Evaluation\n{sep}\n\n")
        desc = {
            'accuracy': ('准确率', 'Accuracy'), 'balanced_accuracy': ('平衡准确率', 'Bal.Acc'),
            'precision': ('精确率', 'Precision'), 'recall': ('召回率', 'Recall'),
            'f1_score': ('F1', 'F1'), 'mcc': ('MCC', 'MCC'), 'kappa': ('Kappa', 'Kappa'),
        }
        for k, v in loocv_metrics.items():
            cn, en = desc.get(k, (k, k))
            f.write(f"    {cn}/{en}: {v:.4f}\n")

        # 3. 测试集
        f.write(f"\n{sep}\n  3. 测试集 / Test Set\n{sep}\n\n")
        desc2 = {
            'accuracy': ('准确率', 'Accuracy'), 'balanced_accuracy': ('平衡准确率', 'Bal.Acc'),
            'precision': ('精确率', 'Precision'), 'recall': ('召回率', 'Recall'),
            'specificity': ('特异度', 'Specificity'), 'npv': ('NPV', 'NPV'),
            'f1_score': ('F1', 'F1'), 'mcc': ('MCC', 'MCC'), 'kappa': ('Kappa', 'Kappa'),
            'log_loss': ('LogLoss', 'LogLoss'), 'brier_score': ('Brier', 'Brier'),
        }
        for k, v in test_metrics.items():
            cn, en = desc2.get(k, (k, k))
            f.write(f"    {cn}/{en}: {v:.4f}\n")
        f.write(f"    ROC AUC: {roc_auc:.4f}\n")
        f.write(f"    AP: {ap:.4f}\n")

        # 4. 分类报告
        f.write(f"\n{sep}\n  4. 分类报告 / Classification Report\n{sep}\n\n")
        f.write(report + "\n")

        # 5. K折
        f.write(f"{sep}\n  5. K折交叉验证 / K-Fold CV\n{sep}\n\n")
        for key, scores in cv_scores.items():
            f.write(f"    {key}: {scores.mean():.4f} ± {scores.std():.4f}\n")
            f.write(f"      各折: {', '.join([f'{s:.4f}' for s in scores])}\n\n")

        # 6. 解读
        f.write(f"{sep}\n  6. 解读 / Guide\n{sep}\n\n")
        for c, d in [("Acc>0.7","可接受/Acceptable"), ("F1>0.7","均衡/Balanced"),
                     ("AUC>0.8","强/Strong"), ("MCC>0.3","优于随机/Better")]:
            f.write(f"    {c:16s} → {d}\n")
        f.write(f"\n{sep}\n  报告结束 / End\n{sep}\n")

    print(f"[✓] 报告已保存：{txt}")