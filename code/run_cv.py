# To perform cross validation using K-folds.

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
from features import get_features1, get_features2

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
        list_logloss: list of log loss values (validation) for each fold
        list_logloss_train: list of log loss values (train set) for each fold
    """
    # Perform k-fold validation
    kf = KFold(X.shape[0], k)

    list_logloss = []
    list_logloss_train = []
    if run_all:
        for i, (train_index, val_index) in enumerate(kf):
            print "   validation set", i+1, "started."
            X_train = X[train_index]
            y_train = y[train_index]
            X_val = X[val_index]
            y_val = y[val_index]
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:,1]
            probs_train = model.predict_proba(X_train)[:,1]
            list_logloss.append(log_loss(y_val,  probs))
            list_logloss_train.append(log_loss(y_train,  probs_train))
    return list_logloss, list_logloss_train


# Main function
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
    #X, y, X_test, indices = get_features2(df_train, target, df_test)
    print "Obtained features as numpy arrays"

    #params_set = [(10,), (50,), (100,), (500,), (1000,), (5000,), (10000,), (50000,)]
    params_set = []
    n_esti = [500, 1000]
    learni = [0.001, 0.5]
    max_de = [3]
    subsam = [1.0]
    max_fe = [None]
    for p1 in n_esti:
        for p2 in learni:
            for p3 in max_de:
                for p4 in subsam:
                    for p5 in max_fe:
                        params_set.append((p1, p2, p3, p4, p5))

    for params in params_set:
        n_estimators = params[0]
        learning_rate = params[1]
        max_depth = params[2]
        subsample = params[3]
        max_features = params[4]
        # Cross validation with given model.
        for i in range(len(X)):
            #model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            model = GradientBoostingClassifier(\
                                learning_rate=learning_rate,\
                                n_estimators=n_estimators,\
                                max_depth=max_depth,\
                                subsample=subsample,\
                                max_features=max_features)
            #model = linear_model.LogisticRegression()
            #print "============================================="
            #print "For set", i+1, "for model", model.__class__.__name__
            list_logloss, list_logloss_train = \
                validate(X[i], y[i], model, k=5, run_all=True)
            #print np.mean(list_logloss_train), list_logloss_train
            params_str = ""
            for p in params:
                params_str += str(p) + " "
            print i+1, params_str, np.mean(list_logloss), list_logloss

    return

if __name__ == '__main__':
    main()
