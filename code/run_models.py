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


# Main func
def main():
    if os.path.exists(xy_pickle_filename):
        with open(xy_pickle_filename, 'r') as f:
            X, y, X_test, indices = pickle.load(f)
        print "Read from the pickled data: " + xy_pickle_filename
    else:
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
        # Storing these arrays as a pickled file.
        with open(xy_pickle_filename, 'wb') as f:
            pickle.dump((X, y, X_test, indices), f)

    # Run models
    y_test_prob = []
    for i in range(len(X)):
        #model = RandomForestClassifier(n_estimators=100, oob_score=True)
        model = linear_model.LogisticRegression()
        list_logloss = validate(X[i], y[i], model, k=5, run_all=True)
        print list_logloss, np.mean(list_logloss)
        print "Running %s with set %d" %(model.__class__.__name__, i+1)
        model = model.fit(X[i], y[i])
        prob_temp = model.predict_proba(X_test[i])[:,1]
        y_test_prob.append(pd.DataFrame(prob_temp, index=indices[i], \
                        columns=['PredictedProb']))

    # combining probabilities from two sets into one for submission.
    # Since IDs were sorted in the original data, probabilities should be sorted
    # by IDs before submission.
    pd.concat([y_test_prob[0], y_test_prob[1]]).sort_index().to_csv( \
        'submission.csv')

    return

if __name__ == '__main__':
    main()
