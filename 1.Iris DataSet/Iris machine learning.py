#用pandas读取sklearn的数据
from sklearn.datasets import load_iris
data = load_iris()
print(dir(data))  # 查看data所具有的属性或方法
print(data.DESCR)  # 查看数据集的简介
import pandas as pd
#直接读到pandas的数据框中
pd.DataFrame(data=data.data, columns=data.feature_names)


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)