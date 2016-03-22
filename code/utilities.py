# Utilities of my own for this problem.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User defined class for error exception
class InvalidDimensionError(Exception):
    pass


# Function to read data frames from csv files.
def read_files(filename_train, filename_test):
    """
    Read csv files for train and test sets
    input:
        filename_train (str): filename for train set (csv)
        filename_test (str): filename for test set (csv)
    output:
        df_train (dataframe): pandas dataframe for the train set
        target (series): pandas series for the target
        df_test (dataframe): pandas dataframe for the test set
    """
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    # The first column is assumed to be ID ('ID' as an index),
    # and the second column is assumed to be target ('target').
    df_train.set_index('ID', inplace=True)
    df_test.set_index('ID', inplace=True)

    # Target is separated from df_train.
    target= df_train.pop('target')

    return df_train, target, df_test


# Function to compare two lists to see if they have the same unique values
def list_compare(list1, list2):
    """
    Fucntion to compare two lists (assumed to have no duplicates)
    input:
        list1, list2: lists
    output:
        boolean value
    """
    set1 = set(list1)
    set2 = set(list2)
    return set1 == set2


# To calcluate the "variability" for a given variable in the train set.
def find_variability(df_train, target, category_variable_name):
    """
    Draw 3 rows of plots (1, Distributions of category variables for the train
    set; 2, Distributions for test sets; 3, target probability for each value
    for the train set)
    input:
        df_train: data frame for the train set
        target : series for the target (assumed to have 0 or 1 only)
        category_variable_name: name of the category variable
    output:
        None (shows plots)
    """
    df = pd.concat([df_train[category_variable_name], target], axis=1)
    target_prob_train = target.mean()
    counts = np.array(pd.value_counts(df[category_variable_name]))
    ratios = np.array(df.groupby(category_variable_name)[target.name].mean())

    variability = (counts * (ratios - target_prob_train)**2).sum()
    return variability/df_train.shape[0]


# Replace extreme probabilities (0 and 1) with values that can be treated.
def replace_extremes(probs):
    """
    input:
        probs: numpy array of floats between 0 and 1
    output:
        probs: numpy array of floats between 0 and 1 (excluding 0 and 1)
    """
    extreme_low = 10**(-15)
    extreme_high = 1 - extreme_low
    for i, p in enumerate(probs):
        if p < extreme_low:
            probs[i] = extreme_low
        elif p > extreme_high:
            probs[i] = extreme_high
    return probs


# The log loss function
def log_loss(target, probs):
    """
    input:
        target: 1d array (series) of 0 and 1
        probs: 1d array(series) of floats between 0 and 1
    output:
        log_loss: value of the log loss function.
    """
    # convert two arrays into numpy arrays first.
    target = np.array(target,dtype='float')
    probs = np.array(probs, dtype='float')

    # check the lengths of two arrays
    if len(target) != len(probs):
        return -1

    # when probs[i]=0 or 1, replace it with 10^(-15) or 1-10^(-15)
    probs = replace_extremes(probs)

    logloss = (target * np.log(probs) + (1 - target) * np.log(1 - probs)).sum()
    return -logloss/len(target)


# Function to plot (1) distributions of category variables
# (for both train and test sets)
# and (2) how much of each value has target 1 in the train set.
# Data frames are assumed to have only predictors as columns and
# target is a separate series with 0 and 1.
def plot_category_variables(df_train, df_test, target, category_variable_names):
    """
    Draw 3 rows of plots (1, Distributions of category variables for the train
    set; 2, Distributions for test sets; 3, target probability for each value
    for the train set)
    input:
        df_train: data frame for the train set
        df_test: data frame for the test set
        target : series for the target (assumed to have 0 or 1 only)
        category_variable_names: list of names of the category variables
    output:
        None (shows plots)
    """
    num_variables_given = len(category_variable_names)
    # Adding target as a column for groupby computations later.
    df_train_category = pd.concat([df_train[category_variable_names], target],\
                                  axis=1)
    df_test_category = df_test[category_variable_names]

    # plotting.
    fig, axs = plt.subplots(3, len(category_variable_names), figsize=(13,10))
    for i, variable_name in enumerate(category_variable_names):
        counts_train = df_train_category.groupby(variable_name).size()
        counts_train.plot(kind='bar', \
                          ax=axs[0,i] if num_variables_given > 1 else axs[0], \
                          color='b', alpha=0.5)
        counts_test = df_test_category.groupby(variable_name).size()
        counts_test.plot(kind='bar', \
                         ax=axs[1,i] if num_variables_given > 1 else axs[1], \
                         color='g', alpha=0.5)
        ratios = df_train_category.groupby(variable_name)[target.name]\
            .mean()
        ratios.plot(kind='bar', ax=axs[2,i] if num_variables_given > 1 else \
                    axs[2], color='r', alpha =0.5)
    plt.suptitle('Total Counts and ratios of target 1')
    plt.show()
    return


# To plot counts and target probabilities of a category variable
# in a 2D scatter plot
def plot_variability(df_train, target, \
                     category_variable_name, binsize=0.1):
    """
    Plots variabilities of category variables.
    input:
        df_train: data frame for the train set
        target : series for the target (assumed to have 0 or 1 only)
        category_variable_name: name of the category variable
        bin_size: size of each bin (uniform size)
    output:
        None (shows plots)
    """
    df = pd.concat([df_train[category_variable_name], target], axis=1)
    target_prob_train = target.mean()
    counts = pd.value_counts(df[category_variable_name])
    ratios = df.groupby(category_variable_name)[target.name].mean()
    plt.scatter(ratios, counts, alpha=0.5)
    plt.xlim((ratios.min() - 0.05, 1.05))
    plt.title(category_variable_name)
    plt.xlabel('target probability')
    plt.ylabel('count')
    plt.axvline(x=target_prob_train, color='r')
    nlimit = int(1.0/binsize + 1)
    for i in range(-nlimit, nlimit+1):
        pos = target_prob_train + i*binsize + binsize/2.0
        if pos >= -binsize and pos <= 1.0 + binsize:
            plt.axvline(x=pos, color='g', ls='dotted')
    plt.show()
    return


# Find the dimensions of train and test sets.
def find_dimensions(df_train, target, df_test):
    """
    input:
        df_train (dataframe): data frame for the train set (ID as an index)
        target (series): response variable for the train set (value: 0 or 1)
        df_test (dataframe): data frame for the test set (ID as an index)
    output:
        num_data_train
        num_data_test
        num_variables_given
    """
    # find dimensions for train & test sets
    num_data_train = df_train.shape[0]
    num_data_test = df_test.shape[0]
    # ID is an index, and target is separate.
    num_variables_given = df_train.shape[1]

    # Check the validity of data sets based on dimensions.
    if df_test.shape[1] != num_variables_given or num_data_train != len(target):
        raise InvalidDimensionError
    return num_data_train, num_data_test, num_variables_given


# Find information for category variables
def find_category_variables(df_train):
    """
    input:
        df_train
    output:
        category_variables (list of column indices)
        numerical_variables (list of column indices)
        unique_values (dict of sets)
    """
    num_variables_given = df_train.shape[1]
    # indices of category variables by looking at types (23 out of 131)
    # Assuming non-category (numerical) variables are in 'float64'
    category_variables = []
    for i in range(df_train.shape[1]):
        if df_train.iloc[:,i].dtype != 'float64':
            category_variables.append(i)

    # The rest of variables are numerical variables.
    numerical_variables = \
        [i for i in range(num_variables_given) if i not in category_variables]

    # Find unique values for category variables in the train set.
    unique_values = {}
    for ind in category_variables:
        unique_values[ind] = set(df_train.iloc[:,ind].unique())

    return category_variables, numerical_variables, unique_values


# Given a category variable, returns a series containing the labels
# (0, 1, 2,..) for each value.
# Labels are found binning the target probability of each value in the train
# set.
def find_binning_info(df_train, target, \
                      category_variable_name, num_bins=10):
    """
    input:
        df_train: data frame for the train set
        target : series for the target (assumed to have 0 or 1 only)
        category_variable_name: name of the category variable
        num_bins: number of bins (default: 10)
    output:
        bin info (series): contains the bin label (int) for each value
    """
    df = pd.concat([df_train[category_variable_name], target], axis=1)
    target_prob_train = target.mean()
    ratios = df.groupby(category_variable_name)[target.name].mean()
    ratio_min = ratios.min()
    binsize = (ratios.max() - ratio_min)/float(num_bins)
    for i in range(-num_bins, num_bins+1):
        pos = target_prob_train + i*binsize + binsize/2.0
        if pos > ratio_min:
            if pos < 1.0 + binsize:
                bin_min = pos - binsize
            else:
                return None # when binsize is not set correctly.
            break

    # bin names are integers
    return pd.Series(pd.cut(ratios, bins=np.arange(bin_min, 1.0 + binsize,\
                    binsize), labels=False), index=ratios.index)


# Find a new column based on conversion information found above
def find_new_variable(column, conversion, unique_values):
    """
    Generate a new variable based on conversion table
    input:
        column (series): a column of a data frame (category)
        conversion (series): conversion table stored as a series.
        unique_values (set): unique values of the given variable.
                            (It can be obtained from conversion table too.)
    output:
        column (series): new column constructed.
    """
    # New column
    col = column.copy()
    for i, v in col.iteritems():
        if v in unique_values: # unique_values: unique values from the train set.
            col[i] = conversion[v]
        else:
            col[i] = conversion['null']
    return col


# Function to add new variables for category variables with too many values.
def add_variables_by_binning(variables_binned, num_bins, df_train, target, \
                             df_test, column_names, unique_values):
    """
    input:
        variables_binned: list of column indices for category variables
        num_bins: number of bins to be made
        df_train (data frame): train set
        target (series): target
        df_test (data frame): test set
        column_names: list of column names for train and test sets
        unique_values: dict of sets of unique values (key: column index)
    output:
        none
    """
    size_binned = len(variables_binned)
    # dict will be elements of this list and each dict contains conversion info.
    conversion_methods = []
    for ind in variables_binned:
        conversion_methods.append(find_binning_info(df_train, target, \
                                column_names[ind], num_bins))

    for i, ind in enumerate(variables_binned):
        df_train[column_names[ind] + 'a'] = \
            find_new_variable(df_train[column_names[ind]], \
                              conversion_methods[i], unique_values[ind])
        df_test[column_names[ind] + 'a'] = \
            find_new_variable(df_test[column_names[ind]], \
                              conversion_methods[i], unique_values[ind])
        column_names.append(column_names[ind] + 'a')
    print "Feature engineering:", size_binned, \
        "category variables are converted to", size_binned, \
        "new features with less categories"
