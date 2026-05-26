"""
main.py  v7.0
决策树专精：分箱 + 穷举特征 + Optuna + Bagging + LOOCV
"""

import config
from data_loader import prepare_raw_data
from feature_engineer import build_all_features, select_best_features
from model_builder import split_data, build_and_tune
from model_evaluator import (
    evaluate_test_set, loocv_evaluate, kfold_evaluate,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_cv_scores, plot_feature_importance, plot_learning_curve
)
from tree_visualizer import visualize_tree
from experiment_saver import (
    create_experiment_dir, save_split_data,
    save_predictions, save_metrics
)


def main():
    save_dir = create_experiment_dir()

    # 1. 加载原始数据
    X_raw, y, raw_df = prepare_raw_data()

    # 2. 特征工程（分箱 + 交互）
    X_all, all_names = build_all_features(X_raw, config.BASE_FEATURES)

    # 3. 穷举特征子集
    best_idx, best_names, subset_f1 = select_best_features(X_all, y, all_names)
    X_selected = X_all[:, best_idx]

    # 4. 划分数据集
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    save_split_data(X_train, X_test, y_train, y_test, best_names, save_dir)

    # 5. 训练决策树
    final_model, best_tree, best_params, all_params = \
        build_and_tune(X_train, y_train, best_names)

    # 6. LOOCV 全数据评估
    if config.USE_LOOCV:
        loocv_pred, loocv_metrics = loocv_evaluate(final_model, X_selected, y)
        plot_confusion_matrix(y, loocv_pred,
                              f'LOOCV 混淆矩阵 (N={len(y)})',
                              save_dir, 'cm_loocv.png')
    else:
        loocv_metrics = {}

    # 7. 测试集评估
    y_pred, y_prob, test_metrics, report = evaluate_test_set(final_model, X_test, y_test)
    save_predictions(X_test, y_test, y_pred, y_prob, best_names, save_dir)
    plot_confusion_matrix(y_test, y_pred,
                          f'测试集混淆矩阵 (N={len(y_test)})',
                          save_dir, 'cm_test.png')

    # 8. ROC / PR
    roc_auc = plot_roc_curve(y_test, y_prob, save_dir)
    ap = plot_pr_curve(y_test, y_prob, save_dir)

    # 9. K 折
    cv_scores = kfold_evaluate(final_model, X_selected, y)
    plot_cv_scores(cv_scores, save_dir)

    # 10. 特征重要性
    plot_feature_importance(best_tree, best_names, save_dir)

    # 11. 学习曲线
    plot_learning_curve(final_model, X_selected, y, save_dir)

    # 12. 决策树可视化
    visualize_tree(best_tree, best_names, save_dir)

    # 13. 保存报告
    save_metrics(test_metrics, loocv_metrics, all_params, cv_scores, report,
                 roc_auc, ap, best_names, subset_f1, save_dir)

    print(f"\n🎉 完成！结果：{save_dir}/")


if __name__ == '__main__':
    main()