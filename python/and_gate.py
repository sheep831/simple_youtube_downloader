from sklearn.linear_model import Perceptron
import numpy as np

# try to learn a AND gate relationship through a perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
Z = np.array([[0, 1], [1, 1], [1, 0], [1, 1]])

# single perceptron is enough for AND gate
model = Perceptron() 
model.fit(X, y)
print(model.predict(Z))