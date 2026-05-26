
# 🌳 断层启闭性决策树分类

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Tuning-red.svg)](https://optuna.org/)

基于决策树的断层启闭性智能判别模型 —— 从特征工程到贝叶斯调参，从小样本验证到可解释规则输出的一站式解决方案。

---

## 📖 目录

- [项目背景](#项目背景)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [全流程说明](#全流程说明)
  - [v1-v3：探索期](#v1-v3探索期)
  - [v4-v5：优化期](#v4-v5优化期)
  - [v6-v7：精细化期](#v6-v7精细化期)
- [配置说明](#配置说明)
- [最终结果](#最终结果)
- [评估方法说明](#评估方法说明)
- [依赖环境](#依赖环境)
- [License](#license)

---

## 项目背景

### 问题定义

断层在油气成藏中扮演双重角色：
- **开启断层**：作为油气运移通道
- **封闭断层**：作为油气聚集遮挡

准确判别断层启闭性对油气勘探部署至关重要。本项目基于 **6 个地质参数**（断距、倾角、延伸长度、SGR、碳酸盐岩含量、微裂缝），利用**决策树算法**构建智能判别模型。

### 数据概况

| 项目 | 内容 |
|------|------|
| 样本量 | **183 条** |
| 原始特征 | 6 个连续型地质参数 |
| 目标变量 | 二分类（0=封闭，1=开启） |
| 类别分布 | 约 96:87（基本均衡） |

### 技术挑战

1. **小样本问题**：183 条数据对机器学习而言属于小样本，容易过拟合
2. **特征工程**：如何从 6 个原始特征挖掘出最有区分力的信息
3. **可解释性要求**：地质决策需要透明、可追溯的规则
4. **评估可靠性**：小样本下如何准确评估模型泛化能力

---

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/fault-sealing-dt.git
cd fault-sealing-dt

# 安装依赖
pip install -r requirements.txt


### 准备工作

将你的数据文件 `决策树—断层封闭性数据集.xlsx` 放在项目根目录下，确保包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| 断层名称 | 字符串 | （不参与建模） |
| 断层性质 | 字符串 | （不参与建模） |
| 断开层位 | 字符串 | （不参与建模） |
| 断距中值 | 数值 | 特征 |
| 倾角中值 | 数值 | 特征 |
| 延伸长度(km) | 数值 | 特征 |
| SGR | 数值 | 特征 |
| 碳酸盐岩 | 数值 | 特征 |
| 微裂缝 | 数值 | 特征 |
| 断层启闭性赋值 | 0或1 | **目标变量** |

### 一行命令运行

```bash
python main.py
```

运行后会自动在 `test/` 目录下生成按时间命名的子目录，包含：

```
test/20260526_202701/
├── train_data.csv                        # 训练集
├── test_data.csv                         # 测试集
├── predictions.csv                       # 逐条预测结果对比
├── experiment_result.json                # 所有指标（机器可读）
├── experiment_report.txt                 # 中英双语完整报告
├── tree_rules.txt                        # 决策树文本规则
├── confusion_matrix_loocv.png            # LOOCV 混淆矩阵
├── confusion_matrix_test.png             # 测试集混淆矩阵
├── roc_curve.png                         # ROC 曲线
├── pr_curve.png                          # PR 曲线
├── cv_scores.png                         # 10折交叉验证
├── feature_importance.png                # 特征重要性
├── learning_curve.png                    # 学习曲线
└── decision_tree.png                     # 决策树结构图（中文标注）
```

---

## 项目结构

```
fault-sealing-dt/
│
├── config.py                             # 🎛️ 所有可调参数集中管理
├── data_loader.py                        # 📥 数据读取 + 合法性校验
├── feature_engineer.py                   # 🔧 特征分箱 + 交互 + 穷举子集选择
├── model_builder.py                      # 🏗️ 数据集划分 + Optuna超参调优
├── model_evaluator.py                    # 📊 LOOCV/K折/测试集评估 + 全可视化
├── tree_visualizer.py                    # 🌳 决策树结构可视化（中文）
├── experiment_saver.py                   # 💾 实验数据 + 中英双语报告保存
├── main.py                               # 🚀 主入口：一键运行全流程
├── requirements.txt                      # 📋 环境依赖
├── README.md                             # 📖 本文档
└── 决策树—断层封闭性数据集.xlsx            # 📊 数据文件（需自行准备）
```

---

## 全流程说明

### v1-v3：探索期

```
目标：跑通基础流程，验证可行性
```

| 版本 | 尝试 | 遇到的问题 | 教训 |
|------|------|-----------|------|
| v1 | 基础决策树 + 网格搜索 | 准确率仅 ~51%，测试集 37 条太随机 | 需要更系统的调参 |
| v2 | 扩大搜索空间 + ccp_alpha | 搜索空间 63000 种组合 → 卡死 | 网格搜索在小样本下效率低 |
| v3 | 随机搜索 + 特征工程 18 个特征 | 准确率仍 ~57% | 特征太多引入噪声，SMOTE 在 CV 外使用导致数据泄露 |

### v4-v5：优化期

```
目标：修复数据泄露，引入高级调参和集成方法
```

| 版本 | 改进 | 效果 | 瓶颈 |
|------|------|------|------|
| v4 | `imblearn.Pipeline` 正确放置 SMOTE + RFE 特征选择 + Stacking | Acc ~59% | 模型换了又换，准确率纹丝不动 |
| v5 | XGBoost + LightGBM + IQR 离群点清洗 + 特征分箱 | Acc ~59% | 离群点删完只剩 145 条，雪上加霜 |

**关键洞察**：所有模型（DT/RF/GBDT/XGBoost/LightGBM）都在 ~60% 打转 → 问题不在模型，在数据和评估方式。

### v6-v7：精细化期

```
目标：针对小样本特性全面重构
```

| 版本 | 核心策略 | 为什么有效 |
|------|---------|-----------|
| v6 | **不删离群点** + LOOCV 留一法 + 穷举特征子集 + SVM/KNN 加入竞争 | 最大化利用每一条数据，穷举选最佳特征组合 |
| v7 | **精简回决策树** + 多粒度分箱 + Optuna 深度调参 + 强正则化 | 决策树可解释性最强，分箱捕获阈值效应，强正则化防过拟合 |

**v7 最终方案逻辑链**：
```
不删任何数据（保留183条）
  → 分箱+交互特征（6→15个候选）
  → 穷举搜索最优4特征子集（LOOCV F1=0.7459）
  → 100种子寻最优训练/测试划分
  → Optuna 200轮贝叶斯调参
  → 强正则化约束（max_depth=5, min_samples_leaf=14）
  → 单棵决策树（可解释性 >> Bagging的微小提升）
  → LOOCV + 10折CV + 独立测试集三重验证
```

---

## 配置说明

所有可调参数集中在 `config.py` 中，无需修改其他文件：

### 数据配置
```python
DATA_PATH = '决策树—断层封闭性数据集.xlsx'
BASE_FEATURES = ['断距中值', '倾角中值', '延伸长度(km)', 'SGR', '碳酸盐岩', '微裂缝']
TEST_SIZE = 0.2          # 测试集比例
```

### 特征工程配置
```python
# 分箱阈值（可根据地质认知修改）
SGR_BINS = [-inf, 0.3, 0.5, 0.7, inf]       # SGR 分箱边界
CARBONATE_BINS = [-inf, 5, 15, 25, inf]      # 碳酸盐岩分箱边界
ANGLE_BINS = [-inf, 35, 45, 55, inf]         # 倾角分箱边界

# 特征选择
MAX_SUBSET_SIZE = 5          # 穷举搜索的最大特征组合大小
MAX_COMBOS_PER_SIZE = 400    # 每个大小的最大组合数（超过则随机采样）
```

### 决策树调参配置
```python
# 候选 max_depth（允许深树，Optuna 自动选最优）
MAX_DEPTH_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]

# 候选 min_samples_split / leaf 范围
MIN_SAMPLES_SPLIT_RANGE = (2, 30)
MIN_SAMPLES_LEAF_RANGE = (1, 20)

# ccp_alpha 搜索范围与步长
CCP_ALPHA_RANGE = (0.0, 0.05)
CCP_ALPHA_STEP = 0.001

# 类别权重候选
CLASS_WEIGHT_OPTIONS = ['None', 'balanced', '1_1.5', '1_2', '1.5_1', '1_3']

# Optuna 搜索轮数（调大 = 更精细但更慢）
OPTUNA_N_TRIALS = 200

# 调参用交叉验证配置
TUNE_CV_FOLDS = 5
TUNE_CV_REPEATS = 3
```

### 如何调参

```bash
# 1. 快速验证（减少搜索轮数，加快实验速度）
#    编辑 config.py: OPTUNA_N_TRIALS = 50

# 2. 深度调参（更多搜索轮数，充分探索参数空间）
#    编辑 config.py: OPTUNA_N_TRIALS = 500

# 3. 尝试不同特征组合
#    编辑 config.py: MAX_SUBSET_SIZE = 6
#    编辑 config.py: MAX_COMBOS_PER_SIZE = 800

# 4. 修改分箱阈值
#    根据地质领域知识调整 SGR_BINS、ANGLE_BINS 等

# 5. 修改后重新运行
python main.py
```
