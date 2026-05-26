"""
tree_visualizer.py  v6.0
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from sklearn.tree import plot_tree, export_text
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def _get_chinese_font():
    for name in ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong',
                 'KaiTi', 'Arial Unicode MS', 'Noto Sans CJK SC']:
        try:
            fp = FontProperties(family=name)
            if fp.get_name():
                return fp
        except:
            continue
    return FontProperties()


def visualize_tree(tree_model, feature_names, save_dir):
    print("\n" + "=" * 55)
    print("          🌳 决策树可视化")
    print("=" * 55)

    depth = tree_model.get_depth()
    n_leaves = tree_model.get_n_leaves()
    print(f"    深度：{depth}，叶节点：{n_leaves}")

    cfp = _get_chinese_font()
    fw = max(24, n_leaves * 3)
    fh = max(12, depth * 3)

    fig, ax = plt.subplots(figsize=(fw, fh))
    plot_tree(tree_model, feature_names=feature_names,
              class_names=['封闭', '开启'],
              filled=True, rounded=True, impurity=True,
              precision=3, ax=ax, fontsize=10)
    for t in ax.texts:
        t.set_fontproperties(cfp)
    ax.set_title('断层启闭性决策树', fontsize=20, fontweight='bold', fontproperties=cfp)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decision_tree.png'), dpi=150, bbox_inches='tight')
    plt.show()

    rules = export_text(tree_model, feature_names=list(feature_names), decimals=3)
    print(f"\n{rules}")
    with open(os.path.join(save_dir, 'tree_rules.txt'), 'w', encoding='utf-8') as f:
        f.write(rules)
    print("=" * 55)