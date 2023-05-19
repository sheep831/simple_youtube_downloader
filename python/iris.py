from sklearn.datasets import load_iris # import iris dataset
from sklearn.neighbors import KNeighborsClassifier  # import KNN classifier
from sklearn.model_selection import train_test_split    # import function to split the dataset into training and testing subsets.
import numpy as np
import joblib

iris = load_iris()
iris_df = iris.data # feature data
target_df = iris.target #  target labels

X_train,X_test,y_train,y_test = train_test_split(
    iris_df,target_df,
    test_size=0.2,
    random_state=np.random.randint(10)) # split the dataset into training and testing subsets. 80% training and 20% testing data 
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train,y_train)


# Generate some new, unseen data for prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.3, 2.9, 5.6, 1.8]])

# Use the trained model to make predictions on the new data
predictions = neigh.predict(new_data)

# Print the predicted class labels
print(predictions)

# Save the trained model
joblib.dump(neigh, 'trained_model.joblib')
