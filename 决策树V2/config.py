"""
config.py
所有可调参数集中管理，修改此文件即可调整实验配置
"""

# ============================================================
#  数据配置
# ============================================================
DATA_PATH = '决策树—断层封闭性数据集.xlsx'

# 原始数值特征列名
BASE_FEATURES = ['断距中值', '倾角中值', '延伸长度(km)', 'SGR', '碳酸盐岩', '微裂缝']

# 目标列名
TARGET_COL = '断层启闭性赋值'

# 测试集比例
TEST_SIZE = 0.1

# 寻找最优划分的随机种子尝试次数
SPLIT_SEED_TRIALS = 100

# ============================================================
#  特征工程配置
# ============================================================

# SGR 分箱阈值（地质意义：<0.3 低封闭，0.3~0.5 中等，0.5~0.7 较高，>0.7 高封闭）
SGR_BINS = [-float('inf'), 0.3, 0.5, 0.7, float('inf')]

# 碳酸盐岩分箱阈值
ITE_BINS = [-float('inf'), 5, 15, 25, float('inf')]

# 倾角分箱阈值
ANGLE_BINS = [-float('inf'), 35, 45, 55, float('inf')]

# 是否使用交叉特征
USE_INTERACTION_FEATURES = True

# 特征选择：最大保留特征数（None 表示不限制，自动选）
MAX_FEATURES_TO_KEEP = None

# 特征选择：LOOCV 穷举的最大特征组合大小
MAX_SUBSET_SIZE = 5

# 穷举时每个大小的最大组合数（超过则随机采样）
MAX_COMBOS_PER_SIZE = 400

# ============================================================
#  决策树调参配置
# ============================================================

# Optuna 搜索轮数
OPTUNA_N_TRIALS = 200

# 候选 max_depth 值
# 关键：允许深树，Optuna 自动通过 ccp_alpha 控制过拟合
MAX_DEPTH_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]

# 候选 min_samples_split 范围
MIN_SAMPLES_SPLIT_RANGE = (2, 30)

# 候选 min_samples_leaf 范围
MIN_SAMPLES_LEAF_RANGE = (1, 20)

# ccp_alpha 搜索范围
CCP_ALPHA_RANGE = (0.0, 0.05)

# ccp_alpha 搜索步长
CCP_ALPHA_STEP = 0.001

# 类别权重候选（字符串标识，代码中解析）
CLASS_WEIGHT_OPTIONS = ['None', 'balanced', '1_1.5', '1_2', '1.5_1', '1_3']

# 调参用的交叉验证配置
TUNE_CV_FOLDS = 5
TUNE_CV_REPEATS = 3

# ============================================================
#  Bagging 配置
# ============================================================

# 是否启用 Bagging
USE_BAGGING = True

# Bagging 中基学习器数量
BAGGING_N_ESTIMATORS = 200

# Bagging 的 max_samples 比例
BAGGING_MAX_SAMPLES = 0.8

# Bagging 的 max_features 比例
BAGGING_MAX_FEATURES = 0.9

# ============================================================
#  评估配置
# ============================================================

# K 折交叉验证折数
KFOLD_CV = 10

# 是否启用 LOOCV 评估
USE_LOOCV = True

# ============================================================
#  输出配置
# ============================================================

# 实验保存根目录
OUTPUT_DIR = 'test'

# 图片 DPI
FIG_DPI = 150