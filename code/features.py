# Functions for feature selections and feature engineering.

import pandas as pd
import numpy as np

from utilities import add_variables_by_binning
from utilities import find_dimensions, find_category_variables

# Function to get features from data frames (feature selections and engineering)
# Here we assume that we will divide data points into two sets,
# which was determined by patterns of missing values.
def get_features1(df_train, target, df_test):
    """
    Make two separate sets: one with all numerical variables and 10 category
    variables, and the other with 8 numerical variables and 10 category
    variables.
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
    category_variables_top10 = [109, 46, 30, 78, 128, 61, 65, 132, 71, 133]
    #category_variables_top10 = [131, 132, 78, 30, 133, 46, 109, 128, 65, 61]
    # Keep in mind that v22, v56, and v113 are replaced by new variables
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
    # For set 1 for both train and test sets
    # =========================================================================
    # Adding train and test sets into one dataframe.
    # We will deal with train and test sets together in one data frame
    df1 = pd.concat([df_train1, df_test1])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only numerical variables
    df1_temp = df1[numerical_variables]
    # imputation
    df1_temp = df1_temp.fillna(df1_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_top10:
        if ind not in variables_binned:
            df_dummy = pd.get_dummies(df1[column_names[ind]],
                                  prefix=column_names[ind])
            df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
            df1_temp = pd.concat([df1_temp, df_dummy], axis=1)

    # Create X12, X1_test, y1 for analysis
    X1 = df1_temp.iloc[:df_train1.shape[0],:].values
    X1_test = df1_temp.iloc[df_train1.shape[0]:,:].values
    y1 = target.loc[df_train1.index].values

    # =========================================================================
    # For set 2 for both train and test sets
    # =========================================================================
    # Find 8 variables (they are the same for both train and test sets)
    variables_less_null = np.array(numerical_variables)\
              [np.where(num_nulls_by_column_train < 0.01 * num_data_train)]

    # Adding train and test sets into one dataframe.
    df2 = pd.concat([df_train2, df_test2])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only 8 numerical variables
    df2_temp = df2[variables_less_null]
    # Imputation
    df2_temp = df2_temp.fillna(df2_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_top10:
        if ind not in variables_binned:
            df_dummy = pd.get_dummies(df2[column_names[ind]],
                                      prefix=column_names[ind])
            df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
            df2_temp = pd.concat([df2_temp, df_dummy], axis=1)

    # Create X2, X2_test, y2 for analysis
    X2 = df2_temp.iloc[:df_train2.shape[0],:].values
    X2_test = df2_temp.iloc[df_train2.shape[0]:,:].values
    y2 = target.loc[df_train2.index].values

    return [X1, X2], [y1, y2], [X1_test, X2_test], \
        [df_test1.index, df_test2.index]

# Function to get features from data frames (feature selections and engineering)
# Here we assume that we will NOT divide data points into several sets.
def get_features2(df_train, target, df_test):
    """
    Only using 8 numerical variables (with not much missing values) and 10
    category variables.
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

    # Saving null locations for train set
    df_train_isnull = df_train.isnull()

    # column names as a list
    column_names = list(df_test.columns)

    # Find category variables and related information
    category_variables, numerical_variables, unique_values = \
        find_category_variables(df_train)

    # Find 8 numerical variables that have less nulls (8 comes from EDA)
    num_nulls_by_column_train = df_train_isnull[numerical_variables].sum(axis=0)
    variables_less_null = np.array(numerical_variables)\
              [np.where(num_nulls_by_column_train < 0.01 * num_data_train)]

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
                                # column_names is assumed to be a list

    # Choosing category variables.
    # Define 10 category variables that will be considered (from EDA)
    #category_variables_considered = [109, 46, 30, 78, 128, 61, 65, 132, 133]
    category_variables_considered = category_variables + [132, 133]

    # Keep in mind that v22, v56, and v113 are replaced by new variables
    # v22->v22a (131), v56->v56a (132), v113->v113a (133)

    # Adding train and test sets into one dataframe.
    df = pd.concat([df_train, df_test])

    # Making dummy variables from category variables.
    # Initially it has only numerical variables
    df_temp = df[variables_less_null]
    # Imputation
    df_temp = df_temp.fillna(df_temp.mean())

    for ind in category_variables_considered:
        if ind not in variables_binned:
            df_dummy = pd.get_dummies(df[column_names[ind]], \
                                  prefix=column_names[ind])
            df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
            df_temp = pd.concat([df_temp, df_dummy], axis=1)

    # Create X, X_test, y for analysis
    X = df_temp.iloc[:df_train.shape[0],:].values
    X_test = df_temp.iloc[df_train.shape[0]:,:].values
    y = target.values

    return [X], [y], [X_test], [df_test.index]


def get_features3(df_train, target, df_test):
    """
    Make two separate sets: one with all numerical variables and all category
    variables, and the other with 8 numerical variables and all category
    variables.
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
    #category_variables_considered = [109, 46, 30, 78, 128, 61, 65, 132, 71, 133]
    category_variables_considered = category_variables + [132, 133]
    # Keep in mind that v22, v56, and v113 are replaced by new variables
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
    # For set 1 for both train and test sets
    # =========================================================================
    # Adding train and test sets into one dataframe.
    # We will deal with train and test sets together in one data frame
    df1 = pd.concat([df_train1, df_test1])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only numerical variables
    df1_temp = df1[numerical_variables]
    # imputation
    df1_temp = df1_temp.fillna(df1_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_considered:
        if ind not in variables_binned:
            df_dummy = pd.get_dummies(df1[column_names[ind]],
                                  prefix=column_names[ind])
            df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
            df1_temp = pd.concat([df1_temp, df_dummy], axis=1)

    # Create X12, X1_test, y1 for analysis
    X1 = df1_temp.iloc[:df_train1.shape[0],:].values
    X1_test = df1_temp.iloc[df_train1.shape[0]:,:].values
    y1 = target.loc[df_train1.index].values

    # =========================================================================
    # For set 2 for both train and test sets
    # =========================================================================
    # Find 8 variables (they are the same for both train and test sets)
    variables_less_null = np.array(numerical_variables)\
              [np.where(num_nulls_by_column_train < 0.01 * num_data_train)]

    # Adding train and test sets into one dataframe.
    df2 = pd.concat([df_train2, df_test2])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only 8 numerical variables
    df2_temp = df2[variables_less_null]
    # Imputation
    df2_temp = df2_temp.fillna(df2_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_considered:
        if ind not in variables_binned:
            df_dummy = pd.get_dummies(df2[column_names[ind]],
                                      prefix=column_names[ind])
            df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
            df2_temp = pd.concat([df2_temp, df_dummy], axis=1)

    # Create X2, X2_test, y2 for analysis
    X2 = df2_temp.iloc[:df_train2.shape[0],:].values
    X2_test = df2_temp.iloc[df_train2.shape[0]:,:].values
    y2 = target.loc[df_train2.index].values

    return [X1, X2], [y1, y2], [X1_test, X2_test], \
        [df_test1.index, df_test2.index]


def get_features4(df_train, target, df_test):
    """
    Make two separate sets: one with all numerical variables and all category
    variables, and the other with 8 numerical variables and all category
    variables.
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
    #category_variables_top10 = [109, 46, 30, 78, 128, 61, 65, 132, 71, 133]
    category_variables_top10 = [131, 132, 78, 30, 133, 46, 109, 128, 65, 61]
    # Keep in mind that v22, v56, and v113 are replaced by new variables
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
    # For set 1 for both train and test sets
    # =========================================================================
    # Adding train and test sets into one dataframe.
    # We will deal with train and test sets together in one data frame
    df1 = pd.concat([df_train1, df_test1])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only numerical variables
    df1_temp = df1[numerical_variables]
    # imputation
    df1_temp = df1_temp.fillna(df1_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_top10:
        df_dummy = pd.get_dummies(df1[column_names[ind]], \
                                  prefix=column_names[ind])
        df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
        df1_temp = pd.concat([df1_temp, df_dummy], axis=1)

    # Create X12, X1_test, y1 for analysis
    X1 = df1_temp.iloc[:df_train1.shape[0],:].values
    X1_test = df1_temp.iloc[df_train1.shape[0]:,:].values
    y1 = target.loc[df_train1.index].values

    # =========================================================================
    # For set 2 for both train and test sets
    # =========================================================================
    # Find 8 variables (they are the same for both train and test sets)
    variables_less_null = np.array(numerical_variables)\
              [np.where(num_nulls_by_column_train < 0.01 * num_data_train)]

    # Adding train and test sets into one dataframe.
    df2 = pd.concat([df_train2, df_test2])

    # For simplicity, we will only consider significant category variables.
    # Initially it has only 8 numerical variables
    df2_temp = df2[variables_less_null]
    # Imputation
    df2_temp = df2_temp.fillna(df2_temp.mean())

    # Creating dummy variables.
    for ind in category_variables_top10:
        df_dummy = pd.get_dummies(df2[column_names[ind]], prefix=column_names[ind])
        df_dummy.drop(df_dummy.columns[:1], axis=1, inplace=True)
        df2_temp = pd.concat([df2_temp, df_dummy], axis=1)

    # Create X2, X2_test, y2 for analysis
    X2 = df2_temp.iloc[:df_train2.shape[0],:].values
    X2_test = df2_temp.iloc[df_train2.shape[0]:,:].values
    y2 = target.loc[df_train2.index].values

    return [X1, X2], [y1, y2], [X1_test, X2_test], \
        [df_test1.index, df_test2.index]
