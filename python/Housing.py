import os
import tarfile
import pandas as pd
from pandas.plotting import scatter_matrix
from six.moves import urllib
import matplotlib.pyplot as plt
from zlib import crc32
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#======================================================================================================================================
# Get the data

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
# 佢會將每個attribute同每一個其他attribute之間嘅關係畫出嚟, 4個attributes就4x4=16個圖
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
housing.dropna(subset=["total_bedrooms"], inplace=True)

# option 2
housing.drop("total_bedrooms", axis=1)

# option 3
median = housing["total_bedrooms"].median() 
# fill missing values with median, directly modify the original data without creating a new DataFrame
housing["total_bedrooms"].fillna(median, inplace=True)

# option 4
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") # create a SimpleImputer instance, strategy: the value to replace missing values
housing_num = housing.drop("ocean_proximity", axis=1) # drop走d有字既attribute
imputer.fit(housing_num) # 將imputer做好, 佢會將每個attribute嘅median存到佢個statistics_ attribute入面
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
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
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
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num) # list of numerical attributes names (無ocean_proximity)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)

