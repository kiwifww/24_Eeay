# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load data
df = pd.read_csv("turkiye-student-evaluation_generic.csv")
print(df.info())
print(df.head())

plt.figure(figsize=(25,15))
sns.heatmap(df.corr(), annot=True)
plt.show()