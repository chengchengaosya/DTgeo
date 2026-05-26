"""
main.py  v6.0
核心策略：不删数据 + 穷举特征子集 + 多模型Optuna + LOOCV + 软投票
"""

from data_loader import prepare_data
from model_builder import split_data, build_and_tune
from model_evaluator import (
    evaluate_test_set, loocv_evaluate_full,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    cross_validate_kfold, plot_cv_scores,
    plot_feature_importance, plot_learning_curve,
    plot_model_comparison
)
from tree_visualizer import visualize_tree
from experiment_saver import (
    create_experiment_dir, save_split_data,
    save_predictions, save_metrics
)


def main():
    save_dir = create_experiment_dir()

    # 1. 数据（不删离群点）
    DATA_PATH = '决策树—断层封闭性数据集.xlsx'
    X, y, feature_names, raw_df = prepare_data(DATA_PATH)

    # 2. 划分
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    save_split_data(X_train, X_test, y_train, y_test, feature_names, save_dir)

    # 3. 训练（穷举特征 + Optuna + 投票）
    (final_model, dt_model, all_params, best_feat_idx,
     best_feat_names, model_results) = build_and_tune(X_train, y_train, feature_names)

    # 4. 特征子集变换
    X_train_sel = X_train[:, best_feat_idx]
    X_test_sel = X_test[:, best_feat_idx]
    X_all_sel = X[:, best_feat_idx]

    # 5. LOOCV 全数据评估（最可靠）
    loocv_pred, loocv_metrics = loocv_evaluate_full(final_model, X_all_sel, y)
    plot_confusion_matrix(y, loocv_pred,
                          f'LOOCV 混淆矩阵 (N={len(y)})',
                          save_dir, 'confusion_matrix_loocv.png')

    # 6. 测试集评估
    y_pred, y_prob, test_metrics, report = evaluate_test_set(
        final_model, X_test_sel, y_test
    )
    save_predictions(X_test_sel, y_test, y_pred, y_prob, best_feat_names, save_dir)
    plot_confusion_matrix(y_test, y_pred,
                          f'测试集混淆矩阵 (N={len(y_test)})',
                          save_dir, 'confusion_matrix_test.png')

    # 7. ROC / PR
    roc_auc = plot_roc_curve(y_test, y_prob, save_dir)
    ap = plot_pr_curve(y_test, y_prob, save_dir)

    # 8. K 折交叉验证
    cv_scores = cross_validate_kfold(final_model, X_all_sel, y, cv=10)
    plot_cv_scores(cv_scores, save_dir)

    # 9. 特征重要性（决策树）
    # 重新在选定特征上训练决策树
    from sklearn.tree import DecisionTreeClassifier
    dt_params = model_results['决策树']['params']
    cw = dt_params.pop('class_weight', None)
    if isinstance(cw, str) and cw not in ['balanced']:
        cw = None
    dt_vis = DecisionTreeClassifier(**dt_params, class_weight=cw, random_state=42)
    dt_vis.fit(X_all_sel[:, :len(best_feat_names)], y)
    plot_feature_importance(dt_vis, best_feat_names, save_dir)

    # 10. 学习曲线
    plot_learning_curve(final_model, X_all_sel, y, save_dir)

    # 11. 多模型对比
    plot_model_comparison(model_results, save_dir)

    # 12. 决策树可视化
    visualize_tree(dt_vis, best_feat_names, save_dir)

    # 13. 保存报告
    save_metrics(
        test_metrics, loocv_metrics, all_params, cv_scores, report,
        roc_auc, ap, best_feat_names, save_dir
    )

    print(f"\n🎉 完成！结果保存至：{save_dir}/")


if __name__ == '__main__':
    main()