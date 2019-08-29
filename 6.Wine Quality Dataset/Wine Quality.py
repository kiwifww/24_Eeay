
# https://www.kaggle.com/booleanhunter/game-of-wines#Section-2:-Exploring-Relationships-between-features
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


# 加载红酒数据集
data = pd.read_csv("winequality-red.csv", sep=';')

# # 显示开头5个记录
# print(data.info())
# print(data.head(n=5))
#
# # 是否有信息缺失
# print(data.isnull().any())


# # 筛选
# n_wines = data.shape[0]
#
# # 有多少酒品质评分超过6分
# quality_above_6 = data.loc[(data['quality'] > 6)]
# n_above_6 = quality_above_6.shape[0]
#
# # 有多少酒品质评分低于5分
# quality_below_5 = data.loc[(data['quality'] < 5)]
# n_below_5 = quality_below_5.shape[0]
#
# # 有多少酒品质评分在5分到6分之间
# quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
# n_between_5 = quality_between_5.shape[0]
#
# # 品质高于6分的酒所占的比例
# greater_percent = n_above_6 * 100 / n_wines
#
# # 打印结果
# print("酒的总数： {}".format(n_wines))
# print("7分以上（含）的酒： {}".format(n_above_6))
# print("不到5分的酒： {}".format(n_below_5))
# print("5分、6分的酒： {}".format(n_between_5))
# print("7分以上（含）的酒所占的比例： {:.2f}%".format(greater_percent))

# # 散点图
# pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde')
# plt.show()

# # 绘制特征间的热图
# correlation = data.corr()
# print(correlation)
# plt.figure(figsize=(14, 12))
# heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
# plt.show()

# # 可视化pH和fixed acidity的关联
# # 创建一个只包含pH和fixed acidity的dataframe
# fixedAcidity_pH = data[['pH', 'fixed acidity']]
#
# # 使用seaborn库基于dataframe初始化joint-grid
# gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, size=6)
#
# # 在网格中绘制回归图
# gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})
#
# # 在网格中绘制分布图
# gridA = gridA.plot_marginals(sns.distplot)
# plt.show()

# fig, axs = plt.subplots(ncols=1, figsize=(10, 6))
# sns.barplot(x='quality', y='volatile acidity', data='volatileAcidity_quality', ax=axs)
# plt.title('quality VS volatile acidity')
# plt.tight_layout()
# plt.show()
# plt.gcf().clear()



# 机器学习
# 定义类别的分割点。
bins = [1, 4, 6, 10]

# 定义类别
quality_labels = [0, 1, 2]
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)

# # 显示头两行
# print(data.head(n=2))
# print(data.info())

# 分离数据为特征和目标标签
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis=1)

from sklearn.model_selection import train_test_split

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_raw, quality_raw,  test_size=0.2,  random_state=0)

# # 显示分离的结果
# print("Training set has {} samples.".format(X_train.shape[0]))
# print("Testing set has {} samples.".format(X_test.shape[0]))

# 从sklearn导入任意3种监督学习分类模型

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 初始化3个模型
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
clf_C = RandomForestClassifier(max_depth=None, random_state=None)

# 计算训练集1%、10%、100%样本数目
samples_100 = len(y_train)
samples_10 = int(len(y_train)*10/100)
samples_1 = int(len(y_train)*1/100)

# 收集结果

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict_evaluate(clf, samples, X_train, y_train, X_test, y_test)