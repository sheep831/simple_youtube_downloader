from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import joblib

# generates synthetic data
# n_features specifies the number of features for each data point (in this case, 2D data points)
X, y = make_blobs(n_samples=5000, n_features=2, centers=15, shuffle=True, random_state=np.random.randint(10))

# Visualize the original clusters
# c and cmap - color map for the scatter plot.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis') # extract the values of the first and second features from the data.
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Clusters')
plt.show()

NUM_CLUSTERS = 5
km = KMeans(
    n_clusters=NUM_CLUSTERS, init='random',
    n_init=10, max_iter=400,
    tol=1e-04, random_state=2
)
y_km = km.fit_predict(X)

# Visualize the clusters after K-means clustering
plt.scatter(X[:, 0], X[:, 1], c=y_km, cmap='viridis')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', s=200, c='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clusters')
plt.show()

print(y_km)

# Save the trained model
joblib.dump(km, 'k_mean_clustering.joblib')
