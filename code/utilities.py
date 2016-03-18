# Function to compare two lists (assumming both lists have unique values)
def list_compare(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1 == set2


# Function to plot distributions of category variables (for train and test sets)
#     and how much of each value has target 1 in the train set.
# data frames are assumed to be with independent variables and
# target is a separate series with 0 and 1.
# It also treats missing values as another value.
def plot_category_variables(df_train, df_test, target, category_variable_names):
    num_variables = len(category_variable_names)
    target_column_name = 'target'
    df_train_category = df_train[category_variable_names].copy()
    df_test_category = df_test[category_variable_names]

    df_train_category[target_column_name] = target
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
        ratios = df_train_category.groupby(variable_name)[target_column_name]\
            .mean()
        ratios.plot(kind='bar', ax=axs[2,i] if num_variables > 1 else axs[2], \
                    color='r', alpha =0.5)
    plt.suptitle('Total Counts and ratios of target 1')
    plt.show()

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


# To plot counts and target probabilities of a category variable
# in a 2D scatter plot
def plot_variability(df_train, target_train, \
                     category_variable_name, binsize=0.1):
    df = pd.concat([df_train[category_variable_name], target_train], axis=1)
    target_prob_train = target_train.mean()
    counts = pd.value_counts(df[category_variable_name])
    ratios = df.groupby(category_variable_name)['target'].mean()
    plt.scatter(ratios, counts, alpha=0.5)
    plt.xlim((ratios.min()-0.05, 1.05))
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


# Given a category variable, returns a series containing the labels
# (0, 1, 2,..) for each value.
# Labels are found binning the target probability of each value in the train
# set.
def find_binning_info(df_train, target_train, variable_name, binsize=0.1):
    df = pd.concat([df_train[variable_name], target_train], axis=1)
    target_prob_train = target_train.mean()
    ratios = df.groupby(variable_name)['target'].mean()
    ratio_min = ratios.min()
    max_num_bins = int(1.0/binsize + 1) # the maximum possible number of bins.
    bin_name = [] # The name of the first bin is 0.
    bin_range = [] # Range of each bin given by a tuple.
    for i in range(-max_num_bins, max_num_bins+1):
        pos = target_prob_train + i*binsize + binsize/2.0
        if pos > ratio_min and pos < 1.0 + binsize:
            bin_min = pos - binsize
            break
    return pd.cut(ratios, bins=np.arange(bin_min, 1.0 + binsize, binsize), \
                  labels=False) # integers


# Find a new column based on conversion information found above
def find_new_variable(column, conversion, unique_values):
    col = column.copy()
    for i, v in col.iteritems():
        if v in unique_values: # unique_values: unique values from the train set.
            col[i] = conversion[v]
        else:
            col[i] = conversion['null']
    return col


