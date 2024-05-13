from sklearn.decomposition import PCA
import numpy as np

def onePC(matrices):
    pca = PCA(n_components=1)
    #flattened = np.vstack([matrix.flatten() for matrix in matrices])
    #cov = np.cov(flattened, rowvar=False)
    flattened = matrices[0]
    for i in range(len(matrices)-1):
        flattened = np.concatenate((flattened, matrices[i+1]), axis=0)
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
    return transformed.flatten()

#pca, EVR = onePC([np.array([[1,2], [2,1], [2,2]]), np.array([[4,5], [5,4], [4,6]]), np.array([[6,5], [7,8], [5,12]])])
#print(EVR)
#print(apply(np.array([1,2]), pca))
#print(apply(np.array([2,2]), pca))