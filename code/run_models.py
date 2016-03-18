# Claim management - Exploratory Analysis

# importing libraries.
import pandas as pd
import numpy as np
import random

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

# import my functions
from utilities import read_files
from utilities import find_binning_info, find_variability, find_new_variable
from features import get_features1

# Reading the train & test sets and produce the data frames
filename_train = '../data/train.csv'
filename_test = '../data/test.csv'

df_train, target, df_test = read_files(filename_train, filename_test)
print "Read csv files"

# Get features from data frames (feature selections and engineering)
#   X: list of sets of predictors (independent variables) from the train set.
#   y: list of sets of targets (response variable) from the test set.
#   X_test: list of sets of predictors (independent variables) from the test set.
#   indices: list of sets of indices
X, y, X_test, indices = get_features1(df_train, target, df_test)
print "Obtained features as numpy arrays"

# Trying to predict the test set (only the first set with less nulls).
y2_test_prob = rf2.predict_proba(X2_test)[:,1]
y2_test_prob = pd.DataFrame(y2_test_prob, index=df_test2.index, \
                            columns=['PredictedProb'])

# combining probabilities from two sets into one for submission.
pd.concat([y1_test_prob, y2_test_prob]).sort_index().to_csv( \
    'test_submission.csv')

