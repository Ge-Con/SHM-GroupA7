from sklearn.decomposition import PCA
import numpy as np

def onePC(matrices):
    pca = PCA(n_components=1)
    flattened = np.vstack([matrix.flatten() for matrix in matrices])
    #cov = np.cov(flattened, rowvar=False)
    pca.fit(flattened)
    EVR = np.sum(pca.explained_variance_ratio_)
    return pca, EVR

def varPC(matrices):
    pca = PCA()
    flattened = np.vstack([matrix.flatten() for matrix in matrices])
    #cov = np.cov(flattened, rowvar=False)
    pca.fit(flattened)

    EVR = np.cumsum(pca.explained_variance_ratio_)
    components = np.argmax(EVR >= 0.95) + 1

    pca = PCA(n_components=components)
    pca.fit(flattened)
    EVR = np.sum(pca.explained_variance_ratio_)
    return pca, EVR

def apply(matrix, pca):
    flattened = matrix.flatten()
    transformed = pca.transform(flattened.reshape(1, -1))
    return transformed.flatten()

#pca, EVR = varPC([np.array([[1,2], [2,2], [2,3]]), np.array([[4,3], [5,4], [4,6]]), np.array([[6,5], [7,8], [5,12]])])
#print(EVR)
#print(apply(np.array([[1,2], [2,3], [3,3]]), pca))
#print(apply(np.array([[1,2], [6,3], [3,4]]), pca))