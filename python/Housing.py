import os
import tarfile
import pandas as pd
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
# print(housing.info()) # get the attributes of the data, no of instances
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

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)