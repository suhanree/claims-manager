# To run models

# importing libraries.
import pandas as pd
import numpy as np
import random
import os
import cPickle as pickle

# importing tools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, cross_val_score

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
from utilities import read_files, log_loss
from utilities import find_binning_info, find_variability, find_new_variable
from features import get_features1

# filenames
filename_train = '../data/train.csv'    # train set
filename_test = '../data/test.csv'      # test set
xy_pickle_filename = '../data/xy.pkl'      # X, y, X_test (pickled)

# Function to do k-fold cross validation
def validate(X, y, model, k=5, run_all=True):
    """
    input:
        df_train (dataframe): train set
        target (series): target variable
        k: number of folds for k-fold (default: 5)
        run_all (bool): True for running all folds, False for running one fold.
    output:
        list_logloss: list of log loss values for each fold
    """
    # Perform k-fold validation
    kf = KFold(X.shape[0], k)

    list_logloss = []
    if run_all:
        for i, (train_index, val_index) in enumerate(kf):
            print "   validation set", i+1, "started."
            model.fit(X[train_index], y[train_index])
            probs = model.predict_proba(X[val_index])[:,1]
            list_logloss.append(log_loss(y[val_index],  probs))
    return np.mean(list_logloss), list_logloss


# Main func
def main():
    # Reading the train & test sets and produce the data frames
    df_train, target, df_test = read_files(filename_train, filename_test)
    print "Read csv files"

    # Get features from data frames (feature selections and engineering)
    # X (list of numpy 2D arrays): list of data points for the train set.
    # y (list of numpy 1D arrays): list of target values for the train set.
    # X_test (list of numpy 2D arrays): list of data points for the test set.
    # indices: list of lists of IDs (representing different sets)
    X, y, X_test, indices = get_features1(df_train, target, df_test)
    print "Obtained features as numpy arrays"

    # Cross validation with given model.
    for i in range(len(X)):
        #model = RandomForestClassifier(n_estimators=100, oob_score=True)
        model = linear_model.LogisticRegression()
        logloss, list_logloss = validate(X[i], y[i], model, k=5, run_all=True)
        print logloss

    return

if __name__ == '__main__':
    main()
