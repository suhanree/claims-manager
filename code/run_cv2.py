# To perform cross validation using K-folds and xgboost.
# Tuning parameters:
# 1. Fix learning_rate, find the best n_estimators (use xgboost cv)
# 2. Tune max_depth & min_child_weight
# 3. Tune gamma
# 4. Find the best n_estimators again.
# 5. Tune subsample * colsample_bytree (max_features)
# 6. Tune regularization parameter (reg_alpha or reg_lambda).
# 7. Go back to 1 and start with another learning_rate.


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
import xgboost as xgb
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.metrics import confusion_matrix, accuracy_score, \
    precision_score, recall_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.grid_search import GridSearchCV

# import my functions
from utilities import read_files, log_loss
from utilities import find_binning_info, find_variability, find_new_variable
from features import get_features1, get_features2, get_features3, get_features4

# filenames
filename_train = '../data/train.csv'    # train set
filename_test = '../data/test.csv'      # test set
xy_pickle_filename = '../data/xy.pkl'      # X, y, X_test (pickled)

# Function to find n_estimators using cv in xgboost
# xgb.cv performs cross validation at each boosting iteration and returns
# the optimum value of n_estimators (number of iterations).
def find_n_estimators(model, X, y, nfold=5, early_stopping_rounds=50):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round=model.get_params()['n_estimators'],
                      nfold=nfold, metrics='logloss',
                      # possible metric: error, rmse, logloss, auc, merror
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=True)
    model.set_params(n_estimators=cvresult.shape[0])
    print "n_estimators:", cvresult.shape[0]
    return

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
    X, y, X_test, indices = get_features3(df_train, target, df_test)
    print "Obtained features as numpy arrays"

    learning_rate=0.05
    n_estimators_list = []
    n_estimators_list.append(500)
    n_estimators_list.append(500)
    skip_find_n_estimators = True
    max_depth=6
    min_child_weight=3
    gamma=0
    subsample=0.9
    colsample_bytree=0.9
    reg_lambda=0.05

    model = []
    for i in range(len(X)):
        model.append(xgb.XGBClassifier(learning_rate=learning_rate,
                            n_estimators=n_estimators_list[i],
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            reg_lambda=reg_lambda))

        if not skip_find_n_estimators:
            find_n_estimators(model[-1], X[i], y[i], nfold=5,
                              early_stopping_rounds=50)

    params_set = []
    max_de = [5,6,7] # max_depth
    min_ch = [2,3,4] # min_child_weight
    gamma_ = [0] # gamma
    subsam = [0.9] # subsample
    colsam = [0.9] # colsample_bytree
    reg_la = [0.005] # reg_lambda
    for p1 in max_de:
        for p2 in min_ch:
            for p3 in gamma_:
                for p4 in subsam:
                    for p5 in colsam:
                        for p6 in reg_la:
                            params_set.append((p1, p2, p3, p4, p5, p6))

    for params in params_set:
        max_depth = params[0]
        min_child_weight = params[1]
        gamma = params[2]
        subsample = params[3]
        colsample_bytree = params[4]
        reg_lambda = params[5]
        # Cross validation with given model.
        for i in range(len(X)):
            model[i].set_params(max_depth=max_depth,
                             min_child_weight=min_child_weight,
                             gamma=gamma,
                             colsample_bytree=colsample_bytree,
                             subsample=subsample,
                             reg_lambda=0,
                             reg_alpha=reg_lambda)
            #model = linear_model.LogisticRegression()
            #print "============================================="
            #print "For set", i+1, "for model", model.__class__.__name__
            list_logloss, list_logloss_train = \
                validate(X[i], y[i], model[i], k=5, run_all=True)
            #print np.mean(list_logloss_train), list_logloss_train
            params_str = "".join(map(lambda x: str(x) + ' ', params))
            print i+1, params_str, np.mean(list_logloss), list_logloss

    return

if __name__ == '__main__':
    main()
