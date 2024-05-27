from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def onePC(matrices):
    """
        Trains PCA for one principal component

        Parameters:
        - matrices (3D numpy array): Training data

        Returns:
        - pca (PCA object): Trained PCA for all components
    """

    #Create PCA object
    pca = PCA()

    #Flatten training matrix to 2D to be able to fit to multiple samples
    flattened = np.vstack([matrix.flatten() for matrix in matrices])
    pca.fit(flattened)

    return pca

def varPC(matrices):
    """
        Trains PCA to keep 95% variance

        Parameters:
        - matrices (3D numpy array): Training data

        Returns:
        - pca (PCA object): Trained PCA to 95% variance
    """

    #Create PCA object
    pca = PCA()

    #Flatten training matrix to 2D to be able to fit to multiple samples
    #Is this the same as is done in one line above?
    flattened = matrices[0]
    for i in range(len(matrices) - 1):
        flattened = np.concatenate((flattened, matrices[i + 1]), axis=0)
    pca.fit(flattened)

    #Calculate explained variance ratio, keep components required for 95% EVR
    EVR = np.cumsum(pca.explained_variance_ratio_)
    components = np.argmax(EVR >= 0.95) + 1

    #Redefine PCA to with correct number of components
    pca = PCA(n_components=components)
    pca.fit(flattened)
    return pca

def apply(list, pca, component=0):
    """
        Applies any PCA model

        Parameters:
        - list (nD numpy array): Test data
        - pca (PCA object): Trained PCA
        - component: Principal component to keep, not specified if model to 95% variance

        Returns:
        - transformed (float): PCA transform of test data
    """

    transformed = pca.transform(list.reshape(1, -1))    #Flatten 2D matrix
    if component != 0:
        transformed = transformed[:, component]         #Select only required component
    return float(transformed.flatten())


def read_matrices_from_folder(dir, freq):
    matrices = []

    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith(f"{freq}_kHz-allfeatures.csv"):  # Assuming files are stored in numpy format

                df = pd.read_csv(os.path.join(root, name))
                matrix = df.values  # Convert DataFrame to numpy array
                matrices.append(matrix)
    return matrices

def doPCA_multiple_Campaigns(train1,train2,train3,train4,test, component=1):
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
        print(EVR)
        list=[]


        for root, dirs, files in os.walk(test):
            for name in files:
                if name.endswith(f"{f}_kHz-allfeatures.csv"):  # Assuming files are stored in numpy format
                    df = pd.read_csv(os.path.join(root, name))
                    matrix = df.values
                    x = apply(matrix, pca, component)
                    list.append(x)
        output.append(list)

    return output