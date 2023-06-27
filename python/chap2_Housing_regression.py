import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#======================================================================================================================================
# Get the data
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing") # datasets\housing
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" 
# https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path): # create datasets\housing
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz") # take housing.tgz path
    urllib.request.urlretrieve(housing_url, tgz_path) # download a file from the specified housing_url and save it locally at the tgz_path
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
 csv_path = os.path.join(housing_path, "housing.csv")
 return pd.read_csv(csv_path)

#======================================================================================================================================
# load the data and take a look at

housing = load_housing_data()
# print(housing.head())
# print(housing.info()) # get the attributes of the data, no of instances, total_bedrooms has some missing values
# print(housing["ocean_proximity"].value_counts()) # how many districts belong to each category
# print(housing.describe())

# housing.hist(bins=50, figsize=(20,15)) # show the histogram of the data
# plt.show()

#======================================================================================================================================
# Create a Test Set
from zlib import crc32

def test_set_check(identifier, test_ratio): # hash the identifier and check if it is less than the test_ratio
 return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test(data, test_ratio):
 shuffled_indices = np.random.permutation(len(data))
 test_set_size = int(len(data) * test_ratio)
 test_indices = shuffled_indices[:test_set_size]
 train_indices = shuffled_indices[test_set_size:]
 return data.iloc[train_indices], data.iloc[test_indices]

def split_train_test_by_id(data, test_ratio, id_column):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
 return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an `index` column

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])

# print(housing["income_cat"].value_counts() / len(housing)) # get the stratified categories and corresponding percentages in the whole dataset

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set)) # get the stratified categories and corresponding percentages in test set

for set_ in (strat_train_set, strat_test_set): # remove the income_cat attribute so the data is back to its original state
 set_.drop("income_cat", axis=1, inplace=True)

#======================================================================================================================================
# Visualize data to gain insights

housing = strat_train_set.copy() # make a copy of training set

# A visualization highlighting high-density areas
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# A better visualization
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
# add a legend to a plot
# plt.legend()

# plt.show()
# we can spot the pattern through the visualization, the housing prices are very much related to the location and population density

#======================================================================================================================================
# Looking for Correlations

# Select only numeric columns from the housing DataFrame
numeric_columns = housing.select_dtypes(include=[np.number])

# Compute the correlation matrix for numeric columns, only linear s (“if x goes up, then y generally goes up/down”)
# -1:negative correlation, 1: positive correlation, 0: no correlation
corr_matrix = numeric_columns.corr()

# get the correlation between median_house_value and other attributes
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function
# 佢會將每個attribute同每一個其他attribute之間嘅關係畫出嚟, 4個attributes就4x4=16個圖from pandas.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#======================================================================================================================================
# try out various attribute combinations

# create new attributes and check the correlation between them and median_house_value
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

numeric_columns = housing.select_dtypes(include=[np.number])
corr_matrix = numeric_columns.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

#======================================================================================================================================
# prepare and preprocess data for ML algorithms
# write functions to do this part !!!
# 1. Data Cleaning

# drop the labels for training set, drop() creates a copy of the data and does not affect strat_train_set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

"""
fix missing features of total_bedrooms
• option 1 : Remove the districts with missing values.
• option 2 : Remove the whole attribute.
• option 3 : Fill the missing values to some value (zero, the mean, the median, etc.)
• option 4 : Use sklearn imputer
"""
# option 1
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# option 2
# housing.drop("total_bedrooms", axis=1)

# option 3
# median = housing["total_bedrooms"].median() 
# fill missing values with median, directly modify the original data without creating a new DataFrame
# housing["total_bedrooms"].fillna(median, inplace=True)

# option 4
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") # create a SimpleImputer instance, strategy: the value to replace missing values
housing_num = housing.drop("ocean_proximity", axis=1) # drop走d有字既attribute
imputer.fit(housing_num) # 將imputer(估計器)做好, 佢會將每個attribute嘅median存到佢個statistics_ attribute入面
"""
The imputer has simply computed the median of each attribute and stored the result in its 
statistics_ instance variable.

print("imputer.statistics_", imputer.statistics_)
print("housing_num.median().values", housing_num.median().values)
both are the same
"""

X = imputer.transform(housing_num) # 將missing values用median填補, return a numpy array
housing_tr = pd.DataFrame(X, columns=housing_num.columns) # convert numpy array to pandas DataFrame

# print(housing_tr.info())

# ----------------------------------------------
# convert text categories to numbers (categorial variable encoding)
# 2. Handling Text and Categorical Attributes

housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# ordinal_encoder.categories_ is a list of categories
# print(ordinal_encoder.categories_)

"""
convert text categories to one-hot vectors so that the categories are in a format that can be understood by these algorithms
not useful in large categories (such as country codes, professions), 
replace them with meaningful data such as a country code could be replaced with the country’s population or GDP per capita
"""
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # output is a SciPy sparse matrix
# print(cat_encoder.categories_) # get the list of categories

# ----------------------------------------------
# custom transformer class
# 3. Custom Transformers
"""
transformers are objects used to preprocess or transform data before feeding it into a machine learning model.
-use to encapsulate data transformation logic
-use for feature engineering, data cleaning, scaling, encoding, etc

Below transformer,  add extra attributes to the data

TransformerMixin is a base class for all transformers in scikit-learn,
it provides fit() and transform() methods.
fit() - performs any necessary computations or setup based on the training data
transform() - applies the transformation to the input data
"""
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # __init__ is the constructor
    # hyperparameter: add_bedrooms_per_room, determine whether to add the bedrooms_per_room attribute or not
    # = False就即係唔用 add_bedrooms_per_room
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    # does nothing and returns self, Python既self即係this, 即個instance
    def fit(self, X, y=None):
        return self
    
    # X is the input dataset, y is the labels
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# ----------------------------------------------
# transformation pipeline
"""
- gathers and integrates various preprocessing methods and steps to prepare and preprocess the data before it is used for training a machine learning model
- can easily apply the same set of preprocessing steps to new data during inference or testing
- the steps could include 
1. data cleaning (handling missing values, removing outliers)
2. feature scaling (normalizing or standardizing numerical features)
3. feature encoding (OneHotEncoder or OrdinalEncoder for categorical features)
4. feature selection (selecting relevant features)
"""

# 處理numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()), # last estimator must be a transformer
 ])

# it calls fit_transform() sequentially on all transformers 總之一條龍做哂所有野
housing_num_tr = num_pipeline.fit_transform(housing_num) 

# 處理numerical + text attributes
# ColumnTransformer used for performing different transformation to different columns
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num) # list of numerical attributes names (無ocean_proximity)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])

# use fit_transform() during training phase
housing_prepared = full_pipeline.fit_transform(housing)

#======================================================================================================================================
# Select a model to train

from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels) # fit the prepared data and the corresponding labels

some_data = housing.iloc[:5] 
some_labels = housing_labels.iloc[:5] 
some_data_prepared = full_pipeline.transform(some_data) 
# print("Predictions:", lin_reg.predict(some_data_prepared)) # Predictions: [210644.6045  317768.8069  210956.4333  59218.9888  189747.5584] 
# print("Labels:", list(some_labels)) # Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]

# measure the root-mean squared error => typical prediction error
from sklearn.metrics import mean_squared_error 
housing_predictions = lin_reg.predict(housing_prepared) 
lin_mse = mean_squared_error(housing_labels, housing_predictions) 
lin_rmse = np.sqrt(lin_mse) # Prediction error is $686xx, underfitting. 1.select a powerful model 2. better data(features) 3. reduce constraints

# ----------------------------------------------
# Train with another model (in this case, DecisionTreeRegressor and RandomForestRegressor)

# Train with DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse) # error is 0.0, overfitting

# Use Cross Validation to split the training set into a smaller training set and a validation set
"""
Scikit-Learn’s K-fold cross-validation
- Cross-validation is a resampling technique that allows you to assess the performance of a model
- randomly splits the training set into 10 distinct subsets called folds
- trains and evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and training on the other 9 folds
"""
from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores) # scores are the opposite of MSE, need to make it +ve before sqrt

def display_cv_scores(scores):
    print("Score:", scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())

# display_cv_scores(tree_rmse_scores)

# try the cross-validation results for linear regression
linreg_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
linreg_rmse_cv = np.sqrt(-linreg_scores)
# display_cv_scores(linreg_rmse_cv)

# ----------------------------------
# Train with RandomForestRegressor
# the rmse and SD of this model is the lowest !
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
# print("forest_rmse", forest_rmse) 

# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores=np.sqrt(-forest_scores)
# display_cv_scores(forest_rmse_scores)

#======================================================================================================================================
# Fine-Tune your model
# 1. tune (find the best) hyperparameters with GridSearchCV
# if data is massive, use randomized search (evaluates a given number of random combinations) instead of grid search
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False, total 12 + 6 = 18 combinations
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# determine the best configuration {'max_features': 8, 'n_estimators': 30}, it will find the configuration with lowest rmse
# the rmse is 49682, which is slightly less than the one not yet tuned (50182, ie the avg of rmse)
# print(grid_search.best_params_) 
# print(grid_search.best_estimator_) # best model or estimator found during the grid search

# ----------------------------------------------
# 2. Then figure out the importance of each attribute
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

# 'median_income' has highest score, and only 'INLAND' of 'ocean proximity' has significant score, so we can drop others
# print(sorted(zip(feature_importances, attributes), reverse=True))

#======================================================================================================================================
# Evaluate Your Model on the Test Set

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1) # input of test set
y_test = strat_test_set["median_house_value"].copy() # labels of test set

# use transform() for applying the pipeline to new, unseen data
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # find the rmse, it is 47730

# Computing 95% confidence interval
# 95% 信心肯定個rmse係within呢個range, $4000 gap可以接受 
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
# [45893.36082829 49774.46796717]

"""
Points to note
- monitor code to check your system’s live performance at regular intervals
- catch performance degradation
- evaluate the system’s input data quality
- train your models on a regular basis using fresh data (automatically or every 6 months)
"""
