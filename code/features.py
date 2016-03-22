# Functions for feature selections and feature engineering.

import pandas as pd
import numpy as np

from utilities import add_variables_by_binning
from utilities import find_dimensions, find_category_variables

# Function to get features from data frames (feature selections and engineering)
# Here we assume that we will divide data points into several sets,
# which was determined by patterns of missing values.
def get_features1(df_train, target, df_test):
    """
    input:
        df_train (dataframe): data frame for the train set (ID as an index)
        target (series): target variable for the train set with value: 0 or 1
        df_test (dataframe): data frame for the test set (ID as an index)
    output:
        X (list of numpy 2D arrays): list of data points for the train set.
        y (list of numpy 1D arrays): list of target values for the train set.
        X_test (list of numpy 2D arrays): list of data points for the test set.
        indices: list of lists of IDs (representing different sets)
    """
    # Find dimensions
    num_data_train, num_data_test, num_variables_given = \
        find_dimensions(df_train, target, df_test)

    # Saving null locations for given data (both train and test sets)
    df_train_isnull = df_train.isnull()
    df_test_isnull = df_test.isnull()

    # column names as a list
    column_names = list(df_test.columns)

    # Find category variables and related information
    category_variables, numerical_variables, unique_values = \
        find_category_variables(df_train)

    # From here on, missing values of category variables will be replaced
    # with a string 'null', which means that missing values will be treated
    # as another category.
    null_str='null'
    df_train.iloc[:, category_variables] = \
        df_train.iloc[:, category_variables].replace(np.nan, null_str)
    df_test.iloc[:, category_variables] = \
        df_test.iloc[:, category_variables].replace(np.nan, null_str)
    #print "Replaced nulls with 'null' for category variables"

    # add new variables for category variables with too many values.
    variables_binned = [21, 55, 112]
    num_bins = 10 # number of bins for each variable
    add_variables_by_binning(variables_binned, num_bins, df_train, target, \
                             df_test, column_names, unique_values)

    # Choosing category variables.
    # Define 10 category variables that will be considered (from EDA)
    category_variables_top10 = [131, 132, 78, 30, 133, 46, 109, 128, 65, 61]
    # v22, v56, v113 are replaced by new variables
    # v22->v22a (131), v56->v56a (132), v113->v113a (133)

    # =========================================================================
    # Choosing 2 sets of data points based on the number of nulls for each row
    # =========================================================================
    # The numbers of cells with nulls (only looking at numerical variables)
    num_nulls_by_row_train = df_train_isnull[numerical_variables].sum(axis=1)
    num_nulls_by_column_train = df_train_isnull[numerical_variables].sum(axis=0)

    num_nulls_by_row_test = df_test_isnull[numerical_variables].sum(axis=1)
    #num_nulls_by_column_test = df_test_isnull[numerical_variables].sum(axis=0)

    # First, we will select rows with more than 80 nulls.
    mask_null_over_80_train = (num_nulls_by_row_train > 80)
    mask_null_over_80_test = (num_nulls_by_row_test > 80)

    # Set with less than 80 nulls (mostly 0 null)
    df_train1 = df_train[~mask_null_over_80_train].copy()
    df_test1 = df_test[~mask_null_over_80_test].copy()
    # Set with more than 80 nulls (mostly 100 nulls)
    df_train2 = df_train[mask_null_over_80_train].copy()
    df_test2 = df_test[mask_null_over_80_test].copy()

    # =========================================================================
    # Imputation
    # =========================================================================
    # We will replace nulls with the mean value of that column
    df_test1_imputed = df_test1.copy()
    df_test1_imputed.fillna(df_test1.mean(), inplace=True)
    df_test2_imputed = df_test2.copy()
    # Here only impute some columns including 8 variables we need.
    # (Some columns have all nulls and no mean value.)
    df_test2_imputed.fillna(df_test2.mean(), inplace=True)
    print "Divided data into two sets"

    # =========================================================================
    # For set 1 for both train and test sets
    # =========================================================================
    df_train1_dropped_na = df_train1.dropna()

    # For simplicity, we will only consider significant category variables.
    # We will deal with train and test sets together in one data frame
    df1 = pd.concat([df_train1_dropped_na, df_test1_imputed])
    # Initially it has only numerical variables
    df1_temp = df1[numerical_variables]
    for ind in category_variables_top10:
        df_dummy = pd.get_dummies(df1[column_names[ind]], \
                                  prefix=column_names[ind])
        df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
        df1_temp = pd.concat([df1_temp, df_dummy], axis=1)

    # Create X12, X1_test, y1 for analysis
    X1 = df1_temp.iloc[:df_train1_dropped_na.shape[0],:].values
    X1_test = df1_temp.iloc[df_train1_dropped_na.shape[0]:,:].values
    y1 = np.array(target.loc[df_train1_dropped_na.index])

    # =========================================================================
    # For set 2 for both train and test sets
    # =========================================================================
    # Find 8 variables (they are the same for both train and test sets)
    variables_less_null = np.array(numerical_variables)\
              [np.where(num_nulls_by_column_train < 0.01 * num_data_train)]

    df_train2_dropped_na = \
        df_train2[list(variables_less_null) + category_variables_top10].dropna()

    # Creating dummy variables. reset_index should be used because ID orders
    # have to be preserved
    df2 = pd.concat([df_train2_dropped_na.reset_index(), \
                     df_test2_imputed[list(variables_less_null) + \
                                      category_variables_top10].reset_index()])
    df2_temp = df2[['ID'] + [column_names[ind] for ind in variables_less_null]]
    for ind in category_variables_top10:
        df_dummy = pd.get_dummies(df2[column_names[ind]], prefix=column_names[ind])
        df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
        df2_temp = pd.concat([df2_temp, df_dummy], axis=1)

    # Create X2, X2_test, y2 for analysis
    X2 = df2_temp.iloc[:df_train2_dropped_na.shape[0],1:].values
    X2_test = df2_temp.iloc[df_train2_dropped_na.shape[0]:,1:].values
    y2 = np.array(target.loc[df_train2_dropped_na.index])

    return [X1, X2], [y1, y2], [X1_test, X2_test], \
        [df_test1.index, df_test2.index]
