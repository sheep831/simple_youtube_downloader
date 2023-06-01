# Load, pre-process and train the iris datasets
import matplotlib.pyplot as plt
import tensorflow as tf

## loading train_dataset
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname='iris_training.csv', origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

## loading test dataset
test_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_dataset_fp = tf.keras.utils.get_file(fname='iris_test.csv', origin=train_dataset_url)
print("Local copy of the test dataset file: {}".format(test_dataset_fp))


## columns name of the train_dataset
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
label_name = 'species'
column_names = feature_names + [label_name]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))