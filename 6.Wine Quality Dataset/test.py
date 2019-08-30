import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
np.random.seed(2017)
titanic = sns.load_dataset("titanic")

sns.barplot(x="sex", y="survived", hue="class", data=titanic)

print(titanic.info())
print(titanic.head(n=5))