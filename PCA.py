from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import warnings
from Interpolating import scale_exact
from sklearn.preprocessing import StandardScaler

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
    """
        Reads matrices from CSV files and saves them to an array of consistent length

        Parameters:
        - dir (str): Root directory of CSV files
        - filename (str): File type of CSV files (no frequency or .csv)
        - freq (str): 3-digit frequency of CSV files (no kHz)
    """

    matrix = []
    rmatrix = []
    state = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == freq + "kHz_" + filename + ".csv":  # Assuming files are stored in numpy format
                df = pd.read_csv(os.path.join(root, name))
                tempmatrix = df.values  # Convert DataFrame to numpy array
                # Flatten
                tempmatrix = tempmatrix.flatten()
                matrix.append(tempmatrix)
                state += 1

    # Scale data to consistent length
    matrix = np.array(matrix).T
    for row in range(len(matrix)):
        rmatrix.append(scale_exact(matrix[row]))
    scaler = StandardScaler()
    rmatrix = scaler.fit_transform(np.array(rmatrix).T).T

    return np.array(rmatrix)

def doPCA_multiple_Campaigns(dir, filename, component=0): #If 0 to 95% var, else expect 1, 2 or 3rd principle component
    """
        Creates and applies either PCA model

        Parameters:
            - dir (string): CSV directory for test and training data
            - components: Principal component to keep, not specified if model to 95% variance

        Returns:
            - output (2 or 3D numpy array): PCA transform of data, for each of 6 frequencies and 30 states (6, 30).
              if to 95% variance each element is an extra dimension of the array containing the transform of unknown length
              if to one principal component each element is an integer
    """

    output = []
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    if component == 0:
        print("PCA to 95% variance")
    else:
        print("--- Component: " + str(component) + " ---")

    for freq in range(6):
        print(frequencies[freq], "kHz")
        list = []
        for testsample in range(5):
            print("Test sample:", testsample+1)
            matrices = []
            for trainsample in range(5):
                if trainsample != testsample:
                    matrices.append(read_matrices_from_folder(dir + "\\" + samples[trainsample], filename, frequencies[freq]))
            #Matrices is list for samples of matrices

            if component == 0:
                pca = varPC(matrices)
            else:
                pca = onePC(matrices, component)

            x = []
            for testsample2 in samples:
                x.append(apply(read_matrices_from_folder(dir + "\\" + testsample2, filename, frequencies[freq]), pca, component))
            list.append(x)

        output.append(list)

    #Removing extra dimension
    return np.array(output)

#dir = "C:\\Users\Jamie\Documents\\Uni\Year 2\Q3+4\Project\MFs"
#print(doPCA_multiple_Campaigns(dir, 1))