# Claims Manager

This is for the problem in Kaggle
([here](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)). 
I am the sole participant in my team.

## Data
The data has 131 features (predictors) excluding ID,
while the train set has 114,321 rows and test set has
114,393 rows. 
The target (label) for the train set is also given as 0 or 1, with the ratio of
1 at 0.761,
which makes this the bianry classification problem.
Features are named from *v1* to *v131*, which means we can only guess
what each feature stands for.
There are 31 category features and the rest are numeric.


### Category features with too many values
Out of 31 category features, 3 features have more than 50 unique values: *v22*
with 18,211, *v56* with 123, and *v125* with 91.
This becomes the problem when we apply ML algorithms, especially tree-based
models, since there are too many possible splits to go over.
Then we have to do something unless we ignore them.
The one method I used here is assign each value a different value based on the ratio of 
target value. For example, for a given value, if the ratio of target 1 is
between 0 and 0.1, it is assigned 0, if between 0.1 and 0.2, it is assigned 1,
and so on.

This method had a problem with *v22* because it has too many values appearing
only once with the ratio either 0 or 1. We know that the standard deviation
of a proportion is proportional to the inverse of the square root of the size
of the sample. 
There is another problem with *v22*: new values, not in the train set, 
appear in the test set (7,367 times, which is 6.44%).
There is no predictive value in these new values, so I had to regard them as
missing values, but there were too many of them.
Also there is not much predictive value in *v22*, either, if we look at
the *variability* of this feature (see 
[EDA](https://github.com/suhanree/claims-manager/blob/master/code/ExploratoryAnalysis.ipynb)
for detail).
So I decided not to use *v22* for the analysis for now.

### Missing values.
Missing values seemed to have patterns.
For category features, about 5% of values are missing, and two features, 
*v30* and *v113*, have majority of them.
For numeric features, about 33% of values are missing, and, except 8 features,
which has less than 1% missing,
about 1/3 of values are missing from each numeric features.
And those mostly come from 45% of rows because 55% of rows don't have any
missing value.
Due to this fact, I decided to divide the data into 2 sets: one with not many
missing values, where I can use most features, and the other with many missing
values, where I can use only 8 numerica features because other numeric features
have too many missing values (55% for the first set and 45% for the second set
for both train and test sets).
See [EDA](https://github.com/suhanree/claims-manager/blob/master/code/ExploratoryAnalysis.ipynb)
for detail.

## Feature selections and feature engineerings
I didn't do much feature selections. For category features, after I look at
feature importances using the random forest, I tried to use top 10 category
features only, but later I used all category features except *v22* with a
little better score.
For numeric features, as mentioned above, I just used 8 numeric features for
the second set, but used all for the first set, only based on missing values.

A lot of feature engineering had to be done to lower the score.
One of the reason is that there exist feature-feature 
nonlinear correlations, but I didn't do much on those.
All I did was dealing with category features with too many data.

## Analysis
Based on the 2-set approach mentioned above, I applied several ML algorithms:
logistic regression, random forest, and gradient boosted trees.
And performed grid search to find the best set of model parameters using 
5-fold cross validation on an EC2 instance.
As expected, gradient boosted trees performed best.
My best score in the public leaderboard was 0.46505, while the winning score
was 0.42233.
I used mostly python sklearn libraries, but for gradient boosted trees, the xgboost
library was much faster than sklearn library with the same results.
