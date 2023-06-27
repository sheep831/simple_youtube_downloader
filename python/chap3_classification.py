import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# MNIST: 70000張高中生handwritten既數字既相, ML既hello world

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser="auto", as_frame = False)

"""
take a look at the keys of dataset => ['data', 'target', 'feature_names', 'DESCR', 'details', 'categories', 'url']
""" 
# mnist.keys()

X, y = mnist["data"], mnist["target"]
# print(X.shape) # (no of samples, features(28x28pixels))(70000, 784)

# Grab one picture and investigate
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()