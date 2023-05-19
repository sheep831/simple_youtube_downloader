import joblib

# Load the trained model
neigh = joblib.load('trained_model.joblib')

# New feature vector
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Modify the values according to your new data

# Use the trained model to predict the class
predictions = neigh.predict(new_data)

# Print the predicted class
print(predictions)
