from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def onePC(matrices):
    pca = PCA(n_components=1)
    flattened = np.vstack([matrix.flatten() for matrix in matrices])
    #cov = np.cov(flattened, rowvar=False)
    #flattened = matrices[0]
    #for i in range(len(matrices)-1):
    #    flattened = np.concatenate((flattened, matrices[i+1]), axis=0)
    pca.fit(flattened)
    EVR = np.sum(pca.explained_variance_ratio_)
    return pca, EVR

def varPC(matrices):
    pca = PCA()
    #flattened = np.vstack([matrix.flatten() for matrix in matrices])
    #cov = np.cov(flattened, rowvar=False)
    flattened = matrices[0]
    for i in range(len(matrices) - 1):
        flattened = np.concatenate((flattened, matrices[i + 1]), axis=0)
    pca.fit(flattened)

    EVR = np.cumsum(pca.explained_variance_ratio_)
    components = np.argmax(EVR >= 0.95) + 1

    pca = PCA(n_components=components)
    pca.fit(flattened)
    EVR = np.sum(pca.explained_variance_ratio_)
    return pca, EVR

def apply(list, pca):
    transformed = pca.transform(list.reshape(1, -1))
    return float(transformed.flatten())
#
# pca, EVR = onePC([np.array([[1,2], [2,1], [2,2]]), np.array([[4,5], [5,4], [4,6]]), np.array([[6,5], [7,8], [5,12]])])
# print(EVR)
# print(apply(np.array([1,2]), pca))
# print(apply(np.array([2,2]), pca))

def read_matrices_from_folder(dir, freq):
    matrices = []

    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith(f"{freq}_kHz-allfeatures.csv"):  # Assuming files are stored in numpy format

                df = pd.read_csv(os.path.join(root, name))
                matrix = df.values  # Convert DataFrame to numpy array
                matrices.append(matrix)
    return matrices

def doPCA_multiple_Campaigns(train1,train2,train3,train4,test):
    # Use the read_matrices_from_folder function to get the matrices from a folder
    output = []
    for freq in range(6):
        if freq == 0:
            f = "050"
        elif freq == 1:
            f = "100"
        elif freq == 2:
            f = "125"
        elif freq == 3:
            f = "150"
        elif freq == 4:
            f = "200"
        elif freq == 5:
            f = "250"

        matrices = []

        #print(f"{f}_kHz-allfeatures.csv")

        for i in range(1,5):
            #matrices.append(read_matrices_from_folder(f"train{i}",f))
            matrices.extend(read_matrices_from_folder(locals()[f"train{i}"], f))
        #print(matrices)
        pca, EVR = onePC(matrices)
        list=[]


        for root, dirs, files in os.walk(test):
            for name in files:
                if name.endswith(f"{f}_kHz-allfeatures.csv"):  # Assuming files are stored in numpy format
                    df = pd.read_csv(os.path.join(root, name))
                    matrix = df.values
                    x = apply(matrix, pca)
                    list.append(x)
                    #print(x)
        output.append(list)
        #print(output)

    return output


# Call the onePC function with the matrices
# pca, EVR = onePC(matrices)

