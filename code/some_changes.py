
# coding: utf-8

# # Claim management - Exploratory Analysis

# ## Importing libraries

# In[34]:

# importing libraries.
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict

# importing tools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split, KFold

# importing models
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, classification_report

# matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from utilities import log_loss


# ## Reading datasets (train and test) and looking at basic information

# In[289]:

# Reading the train & test sets
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')


# In[290]:

# dimensions 
print df_train.shape, df_test.shape

# numbers of rows for train & test sets
num_rows_train = df_train.shape[0]
num_rows_test = df_test.shape[0]

# number of variables
num_variables = df_test.shape[1] - 1 # ID column should not be counted
print num_variables


# If we exclude ID and the target variable, there are 131 variables. As expected, the test set doesn't have the target variable.

# In[291]:

# Check if ID has null and duplicates.
print df_train['ID'].isnull().sum(),     len(df_train['ID'].unique()) == df_train.shape[0]
print df_test['ID'].isnull().sum(),     len(df_test['ID'].unique()) == df_test.shape[0]


# In[292]:

# Set ID as index
df_train.set_index('ID', inplace=True)
df_test.set_index('ID', inplace=True)

# Target is separated from df_train.
target_train = df_train.pop('target')


# In[293]:

# Saving null locations.
df_train_isnull = df_train.isnull()
df_test_isnull = df_test.isnull()


# In[294]:

# column names
column_names = df_test.columns
print column_names


# In[295]:

# The ratio of target value 1
target_prob_train = target_train.mean()
print target_prob_train


# ## Exploring category variables

# In[296]:

# indices of category variables by looking at the data (23 out of 131)
category_variables = [2, 21, 23, 29, 30, 37, 46, 51, 55, 61,
                      65, 70, 71, 73, 74, 78, 90, 106, 109, 111,
                      112, 124, 128]
category_variable_names = column_names[category_variables]

# The rest of variables are numerical variables.
numerical_variables = [i for i in range(num_variables) if i not in category_variables]


# In[302]:

# Find unique values for category variables in train & test sets.
for ind in category_variables:
    print df_train.columns[ind] + ":", df_train.iloc[:,ind].nunique(), df_test.iloc[:,ind].nunique()


# Except the column 'v22' (18,211 values), all category variables have less than 100 values.

# In[298]:

# Function to compare two lists (assumming both lists have unique values)
def list_compare(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1 == set2


# In[299]:

# Checking if unique values are the same for category variables
for i in category_variables:
    print df_train.columns[i], list_compare(df_train.iloc[:,i].unique(),                        df_test.iloc[:,i].unique())


# 6 variables (v22, v47, v56, v71, v79, v113) have different values in train & test sets.

# In[12]:

# There are 4 category variables with integers (v38, v62, v72, v129), and they have no null.
print df_train.isnull().sum(axis=0)[[37, 61, 71, 128]]
print df_test.isnull().sum(axis=0)[[37, 61, 71, 128]]


# In[300]:

# From here on, missing values of category variables will be replaced with a string 'null',
# which means that missing values will be treated as another category.
df_train.iloc[:, category_variables] = df_train.iloc[:, category_variables].replace(np.nan, 'null')
df_test.iloc[:, category_variables] = df_test.iloc[:, category_variables].replace(np.nan, 'null')


# In[303]:

# Find unique values for train & test sets.
unique_values = {}
for ind in category_variables:
    unique_values[ind] = set(df_train.iloc[:,ind].unique())
    #print df_train.columns[ind] + ":", df_train.iloc[:,ind].nunique(), df_test.iloc[:,ind].nunique()


# In[310]:

# If a new value appears in a test set
# (which is not in training set), it will be also replaced by 'null'. 
# This function is called for each row.
def replace_with_null(row, category_variables, unique_values):
    for j, ind in enumerate(category_variables):
        #print i, ind+1, df_test2.iloc[i, ind]
        if row[ind] != 'null' and row[ind] not in unique_values[j]:
            #print i, ind+1, df_test.iloc[i, ind]
            row[ind] = 'null'
    return row


# In[311]:

# To calcluate the above value for a given variable
def find_variability(df_train, target_train, category_variable_name):
    df = pd.concat([df_train[category_variable_name], target_train], axis=1)
    target_prob_train = target_train.mean()
    counts = pd.value_counts(df[category_variable_name])
    ratios = df.groupby(category_variable_name)['target'].mean()
    
    sum = 0
    for val in counts.index:
        sum  = sum + counts[val] * (ratios[val] - target_prob_train)**2
    return sum/df_train.shape[0]


# In[312]:

# print the above value for all category variables.
variabilities = np.zeros(len(category_variable_names))
#print variabilities
for i, name in enumerate(category_variable_names):
    variabilities[i] = find_variability(df_train, target_train, name)
sorted_indices = np.argsort(variabilities)[::-1]
print category_variable_names[sorted_indices], ':', variabilities[sorted_indices]


# ### Create 3 new variables by converting v22, v56, v113 using bins

# In[512]:

# Given a category variable, returns a series containing the labels (0, 1, 2,..) for each value.
# Labels are found binning the target probability of each value in the train set.
def find_binning_info(df_train, target_train, variable_name, binsize=0.1):
    df = pd.concat([df_train[variable_name], target_train], axis=1)
    target_prob_train = target_train.mean()
    # Since the standard deviation of proportions is given by sqrt(p(1-p)/N), 
    # we assume this should be less than binsize to have meaningful value. If N, the number of 
    # occurrences of a given value, is less than this cutoff value, we assign the bin that contains 
    # average (target_prob_train) to the value.
    cutoff_size = target_prob_train*(1.0-target_prob_train)/binsize**2/4
    print variable_name, cutoff_size
    
    counts = pd.value_counts(df[variable_name])
    ratios = df.groupby(variable_name)['target'].mean()
    for k, v in ratios.iteritems():
        if counts[k] <= cutoff_size:
            ratios[k] = target_prob_train
        else:
            if v > 0.95:
                print k, counts[k],  v
    #print ratios, counts
    ratio_min = ratios.min()
    max_num_bins = int(1.0/binsize + 1) # the maximum possible number of bins.
    bin_name = [] # The name of the first bin is 0.
    bin_range = [] # Range of each bin given by a tuple.
    for i in range(-max_num_bins, max_num_bins+1):
        pos = target_prob_train + i*binsize + binsize/2.0
        if pos > ratio_min and pos < 1.0 + binsize:
            bin_min = pos - binsize
            break
    center_bin = int((target_prob_train - bin_min)/binsize)
    #print bin_min, binsize, center_bin 
    cuts = pd.cut(ratios, bins=np.arange(bin_min, 1.0 + binsize, binsize), labels=False) # integers
    
    #for k, v in cuts.iteritems():
    #    if counts[k] <= cutoff_size:
    #        cuts[k] = center_bin
    return cuts, cutoff_size, center_bin


# In[513]:

variables_binned = [21, 55, 112]
conversion_methods = [] # dict will be elements of this list and each dict contains conversion info.
cutoff_sizes = []
center_bins = []

conversion, cutoff_size, center_bin = find_binning_info(df_train, target_train,                                             column_names[variables_binned[0]], binsize=0.05)
conversion_methods.append(conversion)
cutoff_sizes.append(cutoff_size)
center_bins.append(center_bin)
conversion, cutoff_size, center_bin = find_binning_info(df_train, target_train,                                             column_names[variables_binned[1]], binsize=0.05)
conversion_methods.append(conversion)
cutoff_sizes.append(cutoff_size)
center_bins.append(center_bin)
conversion, cutoff_size, center_bin = find_binning_info(df_train, target_train,                                             column_names[variables_binned[2]], binsize=0.05)
conversion_methods.append(conversion)
cutoff_sizes.append(cutoff_size)
center_bins.append(center_bin)
print cutoff_sizes, center_bins


# In[488]:

# Find a new column based on conversion information found above
def find_new_variable(column, conversion, cutoff_size, center_bin, unique_values):
    col = column.copy()
    counts = pd.value_counts(col)
    for i, v in col.iteritems():
        if v in unique_values: # unique_values: unique values from the train set.
            col[i] = conversion[v]
        else:
            if counts[v] < cutoff_size:
                col[i] = center_bin
            else:
                print "New value", v, "with counts", counts[v], "is too significant!"
    #print pd.value_counts(counts)[:10]
    return col


# In[507]:

# Add new columns for 3 variables to both train and test sets.
new_names = []
for i, ind in enumerate(variables_binned):
    df_train[column_names[ind] + 'a'] = find_new_variable(df_train[column_names[ind]],                                     conversion_methods[i], cutoff_sizes[i], center_bins[i], unique_values[ind])
    df_test[column_names[ind] + 'a'] = find_new_variable(df_test[column_names[ind]],                                    conversion_methods[i], cutoff_sizes[i], center_bins[i], unique_values[ind])
    new_names.append(column_names[ind] + 'a')
column_names = np.append(column_names, new_names)
print df_train.shape, df_test.shape, column_names.shape


# In[501]:

def plot_distributions(df_train, df_test, col_name):
    counts_train = pd.value_counts(df_train[col_name])
    x_train = counts_train.index
    y_train = counts_train.values
    counts_test = pd.value_counts(df_test[col_name])
    x_test = counts_test.index
    y_test = counts_test.values
    plt.bar(x_train, y_train, alpha=0.2, color='b')
    plt.bar(x_test+0.5, y_test, alpha=0.2, color='r')
    plt.show()


# In[508]:

plot_distributions(df_train, df_test, 'v22a')


# In[519]:

# Define 10 category variables that will be considered
category_variables_top10 = [131, 132, 78, 30, 133, 46, 109, 128, 65, 61] # category variables to be considered
                            # v22, v56, v113 are replaced by new variables (v22a (131), v56a (132), v113a (133)) 
category_variable_names_new = column_names[category_variables_top10]


# In[520]:

variabilities = np.zeros(len(category_variables_top10))
#print variabilities
for i, ind in enumerate(category_variables_top10):
    variabilities[i] = find_variability(df_train, target_train, column_names[ind])
sorted_indices = np.argsort(variabilities)[::-1]
print variabilities, sorted_indices
# Plotting a bar plot to visualize the result
plt.figure(figsize=(14, 5))
x_locations = np.arange(len(category_variables_top10))# + 0.5
plt.bar(x_locations, variabilities[sorted_indices], align='center')
plt.xticks(range(len(variabilities)), category_variable_names_new[[sorted_indices]])
plt.show()


# In[435]:

# percentages of nulls for only category varialbes
print "For train set:", df_train_isnull.iloc[:, category_variables].sum().sum() /     float(len(category_variables)*num_rows_train)
print "For test set:", df_test_isnull.iloc[:, category_variables].sum().sum() /     float(len(category_variables)*num_rows_test)


# About 5% of cells are nulls for category variables.

# In[436]:

# The percentage of cells with nulls (only looking at numerical variables)
num_nulls_by_row_train = df_train_isnull[numerical_variables].sum(axis=1)
num_nulls_by_column_train = df_train_isnull[numerical_variables].sum(axis=0)
print "For train set:", num_nulls_by_row_train.sum()/float(num_rows_train * num_variables)

num_nulls_by_row_test = df_test_isnull[numerical_variables].sum(axis=1)
num_nulls_by_column_test = df_test_isnull[numerical_variables].sum(axis=0)
print "For test set:", num_nulls_by_row_test.sum()/float(num_rows_test * num_variables)


# ~ 1/3 of cells for numerical variables are nulls for both train & test sets

# In[437]:

# percentage of rows with no null
print "For train set:", sum(num_nulls_by_row_train == 0)/float(num_rows_train)
print "For test set:", sum(num_nulls_by_row_test == 0)/float(num_rows_test)


# Seems like only 55% of rows have no null value for both train & test sets, when we look at only numerical variables. Now we will look into distributions of nulls in more detail.

# In[438]:

# We will do more analysis on missing data.
# First, we will select rows with more than 80 nulls.
mask_null_over_80_train = (num_nulls_by_row_train > 80)
mask_null_over_80_test = (num_nulls_by_row_test > 80)
print np.sum(mask_null_over_80_train), num_rows_train, np.sum(mask_null_over_80_train)/float(num_rows_train)
print np.sum(mask_null_over_80_test), num_rows_test, np.sum(mask_null_over_80_test)/float(num_rows_test)


# ~44% of rows have more than 80 nulls out of 108 values for both sets. If we divide the data into 2 groups based on the number of nulls (80 as the dividing point), you can compare the percentage of 1 for the target. 

# In[439]:

# Percentage of target 1 out of rows with more than 80 rows.
print target_train[mask_null_over_80_train].sum()/float(sum(mask_null_over_80_train))
# Percentage of target 1 out of rows with less than 80 rows.
print target_train[~mask_null_over_80_train].sum()/float(sum(~mask_null_over_80_train))


# They are not significantly different.
# ### Based on columns

# In[440]:

# percentage of columns with less than 1% of nulls
print "For train set:", sum(num_nulls_by_column_train < 0.01 * num_rows_train)
print "For test set:", sum(num_nulls_by_column_test < 0.01 * num_rows_test)


# In[441]:

# Find above 8 variables (they are the same for both train and test sets)
variables_less_null = np.array(numerical_variables)[np.where(num_nulls_by_column_train < 0.01 * num_rows_train)]
print column_names[variables_less_null]


# ## Trying models

# ### Using 8 numerical variables and top 10 category variables (without dividing data)

# In[491]:

#category_variables_top9 = [132, 78, 30, 133, 46, 109, 128, 65, 61]
df_train2 = df_train[list(variables_less_null) + category_variables_top10]
df_test2 = df_test[list(variables_less_null) + category_variables_top10]
df2 = pd.concat([df_train2.reset_index(), df_test2.reset_index()])
print df2.shape


# In[492]:

# Creating dummy variables. reset_index should be used because ID orders have to be preserved

df2_temp = df2[['ID'] + list(column_names[variables_less_null])]
for ind in category_variables_top10:
    df_dummy = pd.get_dummies(df2[column_names[ind]], prefix=column_names[ind])
    #print column_names[ind], df_dummy.shape, df_dummy.isnull().sum().sum()
    df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
    df2_temp = pd.concat([df2_temp, df_dummy], axis=1)
print df2_temp.shape


# In[493]:

df2 = df2_temp.fillna(df2_temp.mean(), inplace=True)


# In[494]:

X = df2.iloc[:df_train2.shape[0],1:].values 
X_test = df2.iloc[df_train2.shape[0]:,1:].values
y = target_train.values
print X.shape, X_test.shape, y.shape


# In[495]:

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1234)
print X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[496]:

rf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf2 = rf2.fit(X_train, y_train)


# In[497]:

# Find the sorted important variables and their importances.
indices_important_variables = np.argsort(rf2.feature_importances_)[::-1]
important_variable_names = df2.columns[indices_important_variables + 1]
importances = rf2.feature_importances_[indices_important_variables]
# Look at above results in a bar chart.
num_var = 20
plt.figure(figsize=(7, 5))
y_locations = np.arange(num_var) + 0.5
plt.barh(y_locations, importances[:num_var], align='center')
plt.yticks(y_locations, important_variable_names[:num_var])
plt.show()


# In[498]:

rf2.oob_score_


# In[455]:

y_val_prob = rf2.predict_proba(X_val)[:,1]
#y_test_prob = pd.DataFrame(y_test_prob, index=df_test.index, columns=['PredictedProb'])
#y_test_prob.to_csv('test_submission.csv')


# In[350]:

# Baseline with 0.5 probs
log_loss(y_val, np.ones(len(y_val))*0.5), log_loss(y_train, np.ones(len(y_train))*0.5)


# In[351]:

# Baseline with 1 probs
log_loss(y_val, np.ones(len(y_val))), log_loss(y_train, np.ones(len(y_train)))


# In[352]:

# Baseline with 0 probs
log_loss(y_val, np.zeros(len(y_val))), log_loss(y_train, np.zeros(len(y_train)))


# In[353]:

# Baseline with random probs between 0 and 1
log_loss(y_val, np.random.uniform(size=len(y_val))), log_loss(y_train, np.random.uniform(size=len(y_train)))


# In[456]:

log_loss(y_val, y_val_prob), log_loss(y_train, rf2.predict_proba(X_train)[:,1])


# In[457]:

rf2 = rf2.fit(X, y)


# In[458]:

rf2.oob_score_


# In[459]:

y_test_prob = rf2.predict_proba(X_test)[:,1]
df_y_test_prob = pd.DataFrame(y_test_prob, index=df_test.index, columns=['PredictedProb'])
df_y_test_prob.to_csv('test_submission.csv')


# ### Using only numerical variables

# In[262]:

df_train3 = df_train[variables_less_null]
df_test3 = df_test[variables_less_null]
df3 = pd.concat([df_train3.reset_index(), df_test3.reset_index()])
df3 = df3.fillna(df3.mean())
print df3.shape
X = df3.iloc[:df_train3.shape[0],1:].values 
X_test = df3.iloc[df_train3.shape[0]:,1:].values
y = target_train.values
print X.shape, X_test.shape, y.shape
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1234)
print X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[263]:

rf3 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf3 = rf3.fit(X_train, y_train)


# In[264]:

rf3.oob_score_


# In[265]:

y_val_prob = rf3.predict_proba(X_val)[:,1]


# In[266]:

log_loss(y_val, y_val_prob), log_loss(y, rf32.predict_proba(X)[:,1])


# In[267]:

rf3 = rf3.fit(X, y)
rf3.oob_score_


# In[268]:

y_test_prob = rf3.predict_proba(X_test)[:,1]
df_y_test_prob = pd.DataFrame(y_test_prob, index=df_test.index, columns=['PredictedProb'])
df_y_test_prob.to_csv('test_submission.csv')


# ## Try more models and feature sets

# In[159]:

df_train[variables_less_null].isnull().sum(),df_test[variables_less_null].isnull().sum()


# In[160]:

df4_train = df_train[variables_less_null].fillna(df_train[variables_less_null].mean())
df4_test = df_test[variables_less_null].fillna(df_train[variables_less_null].mean())


# In[161]:

df4_train.shape, df4_test.shape, target_train.shape


# In[162]:

df4_train2 = df4_train.iloc[:df4_train.shape[0]/2]
df4_val = df4_train.iloc[df4_train.shape[0]/2:]
target_train2 = target_train[:df4_train.shape[0]/2]
target_val = target_train[df4_train.shape[0]/2:]


# In[163]:

df4_train2.shape, df4_val.shape, target_train2.shape, target_val.shape


# In[164]:

X4 = df4_train2.values
X4_val = df4_val.values
y4 = target_train2.values
y4_val = target_val.values
X4_test = df4_test.values


# In[166]:

X4.shape, X4_val.shape, X4_test.shape, y4.shape, y4_val.shape


# In[174]:

#model = linear_model.LogisticRegression()
model = RandomForestClassifier(n_estimators=100, oob_score=True)


# In[175]:

model = model.fit(X4, y4)


# In[176]:

probs = model.predict_proba(X4_val)[:,1]


# In[177]:

log_loss(y4_val, probs), log_loss(y4, lr.predict_proba(X4)[:,1])


# In[158]:

pd.DataFrame(lr.predict_proba(X4_test)[:,1], index=df4_test.index, columns=['PredictedProb']).to_csv('test_submission.csv')


# In[ ]:



