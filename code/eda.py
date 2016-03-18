# Claim management - Exploratory Analysis

# importing libraries.
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict

# importing tools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

# importing models
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.metrics import confusion_matrix, accuracy_score, \
    precision_score, recall_score
from sklearn.metrics import roc_curve, auc, classification_report

# matplotlib
import matplotlib.pyplot as plt

# Reading the train & test sets
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# numbers of rows for train & test sets
num_rows_train = df_train.shape[0]
num_rows_test = df_test.shape[0]

# number of variables
num_variables = df_test.shape[1] - 1 # ID column should not be counted

# Set ID as index
df_train.set_index('ID', inplace=True)
df_test.set_index('ID', inplace=True)

# Target is separated from df_train.
target_train = df_train.pop('target')

# Saving null locations.
df_train_isnull = df_train.isnull()
df_test_isnull = df_test.isnull()

# column names
column_names = df_test.columns

# The ratio of target value 1
target_prob_train = target_train.mean()

# indices of category variables by looking at the data (23 out of 131)
category_variables = [2, 21, 23, 29, 30, 37, 46, 51, 55, 61,
                      65, 70, 71, 73, 74, 78, 90, 106, 109, 111,
                      112, 124, 128]
category_variable_names = column_names[category_variables]

# The rest of variables are numerical variables.
numerical_variables = \
    [i for i in range(num_variables) if i not in category_variables]

# Find unique values for train & test sets.
unique_values = {}
for ind in category_variables:
    unique_values[ind] = set(df_train.iloc[:,ind].unique())

# From here on, missing values of category variables will be replaced
# with a string 'null', which means that missing values will be treated
# as another category.
df_train.iloc[:, category_variables].replace(np.nan, 'null', inplace=True)
df_test.iloc[:, category_variables].replace(np.nan, 'null', inplace=True)


variables_binned = [21, 55, 112]
# dict will be elements of this list and each dict contains conversion info.
conversion_methods = []
conversion_methods.append(find_binning_info(df_train, target_train, \
                        column_names[variables_binned[0]], binsize=0.1))
conversion_methods.append(find_binning_info(df_train, target_train, \
                        column_names[variables_binned[1]], binsize=0.05))
conversion_methods.append(find_binning_info(df_train, target_train, \
                        column_names[variables_binned[2]], binsize=0.05))


new_names = []
for i, ind in enumerate(variables_binned):
    df_train[column_names[ind] + 'a'] = \
        find_new_variable(df_train[column_names[ind]], \
                          conversion_methods[i], unique_values[ind])
    df_test[column_names[ind] + 'a'] = \
        find_new_variable(df_test[column_names[ind]], \
                          conversion_methods[i], unique_values[ind])
    new_names.append(column_names[ind] + 'a')
column_names = np.append(column_names, new_names)

# Define 10 category variables that will be considered
category_variables_top10 = [131, 132, 78, 30, 133, 46, 109, 128, 65, 61]
# v22, v56, v113 are replaced by new variables
# (v22a (131), v56a (132), v113a (133))


# The percentage of cells with nulls (only looking at numerical variables)
num_nulls_by_row_train = df_train_isnull[numerical_variables].sum(axis=1)
num_nulls_by_column_train = df_train_isnull[numerical_variables].sum(axis=0)

num_nulls_by_row_test = df_test_isnull[numerical_variables].sum(axis=1)
num_nulls_by_column_test = df_test_isnull[numerical_variables].sum(axis=0)

# First, we will select rows with more than 80 nulls.
mask_null_over_80_train = (num_nulls_by_row_train > 80)
mask_null_over_80_test = (num_nulls_by_row_test > 80)

# Find above 8 variables (they are the same for both train and test sets)
variables_less_null = np.array(numerical_variables)\
                    np.where(num_nulls_by_column_train < 0.01 * num_rows_train)]
print column_names[vddariables_less_null]

# Set with less than 80 nulls (mostly 0 null)
df_train1 = df_train[~mask_null_over_80_train].copy()
df_test1 = df_test[~mask_null_over_80_test].copy()
# Set with more than 80 nulls (mostly 100 nulls)
df_train2 = df_train[mask_null_over_80_train].copy()
df_test2 = df_test[mask_null_over_80_test].copy()

# We will replace nulls with the mean value of that column
df_test1_imputed = df_test1.copy()
df_test1_imputed.fillna(df_test1.mean(), inplace=True)
df_test2_imputed = df_test2.copy()
# Here only impute some columns including 8 variables we need.
# (Some columns have all nulls and no mean value.)
df_test2_imputed.fillna(df_test2.mean(), inplace=True)

# To see the feature importance, we will drop rows with null values
# in the numerical variables in the set 1.
# To do that, we fill 'null' in the missing values of category variables first
# (which is done already).
df_train1_dropped_na = df_train1.dropna()

# For simplicity, we will only consider significant
# (and comparably treatable) category variables.
df1 = pd.concat([df_train1_dropped_na, df_test1_imputed])
df1_temp = df1[numerical_variables] # Initially it has only numerical variables
for ind in category_variables_top10:
    df_dummy = pd.get_dummies(df1[column_names[ind]], prefix=column_names[ind])
    #print column_names[ind], df_dummy.shape, df_dummy.isnull().sum().sum()
    df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
    df1_temp = pd.concat([df1_temp, df_dummy], axis=1)

# Create X12, X1_test, y1 for analysis
X1 = df1_temp.iloc[:df_train1_dropped_na.shape[0],:].values
X1_test = df1_temp.iloc[df_train1_dropped_na.shape[0]:,:].values
y1 = np.array(target_train.loc[df_train1_dropped_na.index])

# Using the random forest to find the feature importances
# for numerical variables + 7 category variables.
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf1 = rf1.fit(X1, y1)


# Find the sorted important variables and their importances, and look at top 10.
indices_important_variables12 = np.argsort(rf12.feature_importances_)[::-1]
important_variable_names12 = df1_temp.columns[indices_important_variables12]
importances12 = rf12.feature_importances_[indices_important_variables12]

print rf1.oob_score_

# Trying to predict the test set (only the first set with less nulls).
y1_test_prob = rf1.predict_proba(X1_test)[:,1]
y1_test_prob = pd.DataFrame(y1_test_prob, index=df_test1.index, \
                            columns=['PredictedProb'])


df_train2_dropped_na = \
    df_train2[list(variables_less_null) + category_variables_top10].dropna()

# Creating dummy variables. reset_index should be used because ID orders
# have to be preserved
df2 = pd.concat([df_train2_dropped_na.reset_index(), \
                 df_test2_imputed[list(variables_less_null) + \
                                  category_variables_top10].reset_index()])
df2_temp = df2[['ID'] + list(column_names[variables_less_null])]
for ind in category_variables_top10:
    df_dummy = pd.get_dummies(df2[column_names[ind]], prefix=column_names[ind])
    df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
    df2_temp = pd.concat([df2_temp, df_dummy], axis=1)

# Create X2, X2_test, y2 for analysis
X2 = df2_temp.iloc[:df_train2_dropped_na.shape[0],1:].values
X2_test = df2_temp.iloc[df_train2_dropped_na.shape[0]:,1:].values
y2 = np.array(target_train.loc[df_train2_dropped_na.index])

rf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf2 = rf2.fit(X2, y2)

# Find the sorted important variables and their importances.
indices_important_variables2 = np.argsort(rf2.feature_importances_)[::-1]
important_variable_names2 = \
    df_train2_dropped_na2.columns[indices_important_variables2]
importances2 = rf2.feature_importances_[indices_important_variables2]

print rf2.oob_score_

# Trying to predict the test set (only the first set with less nulls).
y2_test_prob = rf2.predict_proba(X2_test)[:,1]
y2_test_prob = pd.DataFrame(y2_test_prob, index=df_test2.index, \
                            columns=['PredictedProb'])

# combining probabilities from two sets into one for submission.
pd.concat([y1_test_prob, y2_test_prob]).sort_index().to_csv( \
    'test_submission.csv')

