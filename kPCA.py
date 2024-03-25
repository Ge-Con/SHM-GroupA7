from sklearn.decomposition import KernelPCA

def kPCA(X):
    kPCA = KernelPCA
    X_trans = kPCA.fit_transform(X)
