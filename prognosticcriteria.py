import numpy as np

def Pr(X):

    M = len(X)
    Nfeatures = X.shape[1]
    top = np.zeros((M,Nfeatures))
    bottom = np.zeros((M,Nfeatures))

    for j in range(M):
        top[j,:] = X[j,-1]
        bottom[j,:] = np.abs(X[j,0]-X[j,-1])

    prognosability = np.exp(-np.std(top)/np.mean(bottom))

    return prognosability
