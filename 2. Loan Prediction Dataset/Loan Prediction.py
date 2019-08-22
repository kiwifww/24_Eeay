import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',1000)
df = pd.read_csv('train.csv')

# #1.Quick Data Exploration
# print(df.info())
# print(df.head(10))
# print(df.describe())
# print(df['Property_Area'].value_counts())

#2.Distribution analysis
# print(df['Property_Area'].value_counts())
# print(df.groupby('Property_Area').count())

plt.subplot(221)
p1 = df['ApplicantIncome'].hist(bins=25)
plt.subplot(222)
p2 = df.boxplot(column='ApplicantIncome')
plt.subplot(223)
p3 = df['LoanAmount'].hist(bins=25)
plt.subplot(224)
p4 = df.boxplot(column='LoanAmount')
plt.show()
#
# #3.Categorical variable analysis
#
# temp1 = df['Credit_History'].value_counts()
# print ('Frequency Table for Credit History:')
# print (temp1)
# temp2 = df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
# print ('\nProbility of getting loan for each Credit History class:')
# print (temp2)
# temp3 = df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
# print ('\n')
#
# fig = plt.figure(figsize=(8,4))
# ax1 = fig.add_subplot(121)
# temp1.plot(kind='bar')
#
# ax2 = fig.add_subplot(122)
# plt.bar(temp2.index, temp2['Loan_Status'])
#
# temp3 = pd.crosstab([df['Credit_History'],df['Gender']], df['Loan_Status'])
# temp3.plot(kind='bar', stacked=True, grid=False)
# plt.show()


# # 5. Building a Predictive Model in Python
# df['Gender'].fillna(df['Gender'].value_counts().idxmax(), inplace=True)
# df['Married'].fillna(df['Married'].value_counts().idxmax(), inplace=True)
# df['Dependents'].fillna(df['Dependents'].value_counts().idxmax(), inplace=True)
# df['Self_Employed'].fillna(df['Self_Employed'].value_counts().idxmax(), inplace=True)
# df["LoanAmount"].fillna(df["LoanAmount"].mean(skipna=True), inplace=True)
# df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
# df['Credit_History'].fillna(df['Credit_History'].value_counts().idxmax(), inplace=True)
#
# from sklearn.preprocessing import LabelEncoder
# var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
# le = LabelEncoder()
# for i in var_mod:
#   df[i] = le.fit_transform(df[i])
# df.dtypes
#
# # Import models from scikit learn module:
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold  # For K-fold cross validation
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn import metrics
#
#
# # Generic function for making a classification model and accessing performance:
# def classification_model(model, data, predictors, outcome):
#   # Fit the model:
#   model.fit(data[predictors], data[outcome])
#
#   # Make predictions on training set:
#   predictions = model.predict(data[predictors])
#
#   # Print accuracy
#   accuracy = metrics.accuracy_score(predictions, data[outcome])
#   print("Accuracy : %s" % "{0:.3%}".format(accuracy))
#
#   # Perform k-fold cross-validation with 5 folds
#   kf = KFold(5,shuffle=False)
#   error = []
#   for train, test in kf.split(data):
#     # Filter training data
#     train_predictors = (data[predictors].iloc[train, :])
#
#     # The target we're using to train the algorithm.
#     train_target = data[outcome].iloc[train]
#
#     # Training the algorithm using the predictors and target.
#     model.fit(train_predictors, train_target)
#
#     # Record error from each cross-validation run
#     error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))
#
#   print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
#
#   # Fit the model again so that it can be refered outside the function:
#   model.fit(data[predictors], data[outcome])
#
# outcome_var = 'Loan_Status'
# model = RandomForestClassifier(n_estimators=100)
# predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
# classification_model(model, df, predictor_var, outcome_var)
#
# #Create a series with feature importances:
# featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
# print (featimp)