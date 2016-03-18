# Utilities of my own for this problem.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    counts = np,array(pd.value_counts(df[category_variable_name]))
    ratios = np.array(df.groupby(category_variable_name)[target.name].mean())

    variability = (counts * (ratios[val] - target_prob_train)**2).sum()
    return variability/df_train.shape[0]


# Given a category variable, returns a series containing the labels
# (0, 1, 2,..) for each value.
# Labels are found binning the target probability of each value in the train
# set.
def find_binning_info(df_train, target, \
                      category_variable_name, binsize=0.1):
    """
    input:
        df_train: data frame for the train set
        target : series for the target (assumed to have 0 or 1 only)
        category_variable_name: name of the category variable
        bin_size: size of each bin (assumed to be uniform)
    output:
        bin info (series): contains the bin label (int) for each value
    """
    df = pd.concat([df_train[variable_name], target], axis=1)
    target_prob_train = target.mean()
    ratios = df.groupby(variable_name)[target.name].mean()
    ratio_min = ratios.min()
    max_num_bins = int(1.0/binsize + 1) # the maximum possible number of bins.
    for i in range(-max_num_bins, max_num_bins+1):
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
    num_variables = len(category_variable_names)
    # Adding target as a column for groupby computations later.
    df_train_category = pd.concat([df_train[category_variable_names], target],\
                                  axis=1)
    df_test_category = df_test[category_variable_names]

    # plotting.
    fig, axs = plt.subplots(3, len(category_variable_names), figsize=(13,10))
    for i, variable_name in enumerate(category_variable_names):
        counts_train = df_train_category.groupby(variable_name).size()
        counts_train.plot(kind='bar', \
                          ax=axs[0,i] if num_variables > 1 else axs[0], \
                          color='b', alpha=0.5)
        counts_test = df_test_category.groupby(variable_name).size()
        counts_test.plot(kind='bar', \
                         ax=axs[1,i] if num_variables > 1 else axs[1], \
                         color='g', alpha=0.5)
        ratios = df_train_category.groupby(variable_name)[target.name]\
            .mean()
        ratios.plot(kind='bar', ax=axs[2,i] if num_variables > 1 else axs[2], \
                    color='r', alpha =0.5)
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
