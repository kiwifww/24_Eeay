import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("Heights and Weights Dataset.csv")
df = df.iloc[:, [1,2]]
# print(df.info())
# print(df.describe())
xValue = df.iloc[:, [0]]
yValue = df.iloc[:, [1]]
plt.scatter(xValue, yValue, marker=".")
plt.show()

model = LinearRegression()
model.fit(xValue,yValue)