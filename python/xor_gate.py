from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y1 = np.array([0, 1, 0, 0])
y2 = np.array([0, 0, 1, 0])
y3 = np.array([0, 1, 1, 1]) # OR gate checking

layer1_1 = Perceptron()
layer1_2 = Perceptron()
layer2 = Perceptron()

layer1_1.fit(X, y1)
layer1_2.fit(X, y2)
layer2.fit(X, y3)

# layer1_1 outputs
layer1_1_out = layer1_1.predict(X)
layer1_1_out = layer1_1_out.reshape(-1, 1) # reshape to 4x1

# layer1_2 outputs
layer1_2_out = layer1_2.predict(X)
layer1_2_out = layer1_2_out.reshape(-1, 1)

# form layer2 inputs
layer2_in = np.concatenate((layer1_1_out, layer1_2_out), axis=1)

# final result
print(layer2.predict(layer2_in))