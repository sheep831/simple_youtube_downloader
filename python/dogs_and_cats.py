import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# Specify the Kaggle dataset name
dataset_name = 'tongpython/cat-and-dog'

# Specify the destination folder to save the dataset
destination_folder = 'C:\\Users\\Apex\\Desktop\\test_python_nodejs\\python\\cats_and_dogs'

# # Initialize the Kaggle API
# api = KaggleApi()
# api.authenticate()

# # Download the dataset
# api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)

# Define the parent directory containing the subdirectories of images
training_dir = os.path.join(destination_folder, 'training_set', 'training_set')
test_dir = os.path.join(destination_folder, 'test_set', 'test_set')

# Load images and labels
images = []
labels = []

# Define the target size for resizing the images
target_size = (100, 100)

for label, class_name in enumerate(["cats", "dogs"]):
    # Training set
    class_dir = os.path.join(training_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        if image is not None: # to avoid the _DS_Store file
            image = cv2.resize(image, target_size)
        images.append(image)
        labels.append(label)

    # Test set
    class_dir = os.path.join(test_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, target_size)
        images.append(image)
        labels.append(label)


# Filter out None images
valid_indices = [i for i, img in enumerate(images) if img is not None] # to remove the _DS_Store file
images = [images[i] for i in valid_indices]
labels = [labels[i] for i in valid_indices]

# Convert lists to NumPy arrays
# to facilitate further processing and compatibility with other libraries and algorithms
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
neigh = KNeighborsClassifier(n_neighbors=20)

# Reshape the training data to flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)

neigh.fit(X_train_flat, y_train)


# Reshape the test data
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Evaluate the model on the test set
accuracy = neigh.score(X_test_flat, y_test)
print("Accuracy:", accuracy)
