import pandas as pd
import numpy as np
from main import list_files
import csv
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target
labels = np.reshape(breast_labels, (569,1))

final_breast_data = np.concatenate([breast_data, labels], axis=1)
breast_dataset = pd.DataFrame(final_breast_data)

def PCA(X, num_components):

    #1. Finding normalized matrix

    #2. Finding covariance matrix
    covX = np.cov(X, rowvar=False)
    print(pd.DataFrame.cov())
    #3. Finding eigenvectors+values
    eigenvalues,  eigenvectors = np.linalg.eig(covX)

    #4. Sort eigenvectors+values
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    print("------")
    PC = eigenvectors[:num_components]

    #5. Transform original data
    X_transf = PC.fit_transform(X)

PCA(final_breast_data,2)