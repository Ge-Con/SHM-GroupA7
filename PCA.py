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
    stacked = np.concatenate([matrix for matrix in matrices])
    pca.fit(stacked)

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
    stacked = np.concatenate([matrix for matrix in matrices])
    pca.fit(stacked)

    #Calculate explained variance ratio, keep components required for 95% EVR
    EVR = np.cumsum(pca.explained_variance_ratio_)
    components = np.argmax(EVR >= 0.95) + 1

    #Redefine PCA to with correct number of components
    pca = PCA(n_components=components)
    pca.fit(stacked)
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

    transformed = []
    for state in range(30):
        transformed.append(pca.transform(np.array([list[state]])))    #Flatten 2D matrix
    if component != 0:
        for state in range(30):
            transformed[state] = transformed[state][:, component-1][0]         #Select only required component
    else:
        for state in range(30):
            transformed[state] = transformed[state][:][0]   #Removing extra dimension
    return transformed


def read_matrices_from_folder(dir, filename, freq):
    matrix = np.zeros((139, 30))

    for root, dirs, files in os.walk(dir):
        for name in files:

            if name == freq + "kHz_" + filename + ".csv":  # Assuming files are stored in numpy format

                df = pd.read_csv(os.path.join(root, name))
                tempmatrix = df.values  # Convert DataFrame to numpy array
                # Interpolate
                tempmatrix = tempmatrix.T
                count = 0
                for row in range(len(tempmatrix)):
                    matrix[row] = scale_exact(tempmatrix[row])
                    count += 1
    return np.array(matrix)

def doPCA_multiple_Campaigns(dir, component=0): #If 0 to 95% var, else expect 1, 2 or 3rd principle component
    # Use the read_matrices_from_folder function to get the matrices from a folder

    """
    Creates and applies any PCA model

    Parameters:
        - dir (string): CSV directory for test and training data
        - components: Principal component to keep, not specified if model to 95% variance

    Returns:
        - output (2 or 3D numpy array): PCA transform of data, for each of 6 frequencies and 30 states (6, 30).
          if to 95% variance each element is an extra dimension of the array containing the transform of unknown length
          if to one principal component each element is an integer
    """

    output = []
    matrices = []
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["L1-03", "L1-04", "L1-05", "L1-09", "L1-23"]
    filename = "MF"


    for freq in range(6):
        for testsample in range(5):
            for trainsample in range(5):
                if trainsample != testsample:
                    matrices.append(read_matrices_from_folder(dir + "\\" + samples[trainsample], filename, frequencies[freq]))
            #Matrices is list for samples of matrices

            if component == 0:
                pca = varPC(matrices)
            else:
                pca = onePC(matrices, component)

            list = []
            x = apply(matrices[testsample], pca, component)
            list.append(x)

        output.append(list)

    #Removing extra dimension
    return np.array(output)[:, 0]

#dir = "C:\\Users\Jamie\Documents\\Uni\Year 2\Q3+4\Project\MFs"
#print(doPCA_multiple_Campaigns(dir, 0))