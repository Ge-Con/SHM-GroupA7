from sklearn.decomposition import KernelPCA
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def kPCA(X, n, kernel, gamma):
    kPCA = KernelPCA(n_components=n, kernel=kernel, gamma=gamma)
    # How to determine ideal gamma?
    X_trans = kPCA.fit_transform(X)

    column_names = [f"principal component {i+1}" for i in range (n)]
    X_Df = pd.DataFrame(data = X_trans, columns = column_names)

    plt.title("Kernel Principal Component Analysis")
    plt.scatter(X_trans[:,0], X_trans[:,1])
    plt.show()
    return X_Df

from sklearn.datasets import make_moons
X,y = make_moons(n_samples=500, noise = 0.02, random_state = 417)
print(kPCA(X, 2, "rbf", 15))