from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import warnings
from Interpolating import scale_exact

warnings.filterwarnings('ignore')

def onePC(matrices, component):
    """
        Trains PCA for one principal component

        Parameters:
        - matrices (3D numpy array): Training data

        Returns:
        - pca (PCA object): Trained PCA for all components
    """

    #Create PCA object
    pca = PCA(n_components=component)

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
        transformed = transformed[:, component-1]         #Select only required component
    return float(transformed.flatten())


def read_matrices_from_folder(dir, filename, freq):
    matrices = []
    matrix = np.zeros((139, 30))

    for root, dirs, files in os.walk(dir):
        for name in files:

            if name == freq + "kHz_" + filename + ".csv":  # Assuming files are stored in numpy format

                df = pd.read_csv(os.path.join(root, name))
                tempmatrix = df.values  # Convert DataFrame to numpy array
                # Interpolate
                tempmatrix = tempmatrix.T
                for row in range(len(tempmatrix)):
                    matrix[row] = scale_exact(tempmatrix[row])
                matrices.append(matrix)
    return np.array(matrices)

def doPCA_multiple_Campaigns(dir, component=1):
    # Use the read_matrices_from_folder function to get the matrices from a folder
    output = []
    matrices = []
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["L1-03", "L1-04", "L1-05", "L1-09", "L1-23"]
    filename = "MF"

    for freq in range(6):
        for sample in range(5):
            matrices.append(read_matrices_from_folder(dir + "\\" + samples[sample], filename, frequencies[freq]))
        #Matrices is list for samples of lists of matrices

        pca = onePC(matrices, component)
        list = []

        for sample in range(5):
            x = apply(matrices[sample], pca, component)
            list.append(x)

        output.append(list)

    for i in range(len(output)):
        output[i] = output[i][0]

    return output