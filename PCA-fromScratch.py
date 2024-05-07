import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

#breast = load_breast_cancer()
#breast_data = breast.data
#breast_labels = breast.target
#labels = np.reshape(breast_labels, (569,1))

#final_breast_data = np.concatenate([breast_data, labels], axis=1)
#breast_dataset = pd.DataFrame(final_breast_data)
#print(final_breast_data)

def PCA(X, n): # n is the number of principal components

    #TODO: Lines 17–19: For normalization, whether min-max or zero-mean, you should define an input
    # flag (variable) for the function, where we can determine whether we want to do so or not.
    # In the case of HI construction or RUL prediction, you should be aware that we are not allowed
    # to use future data as they are not available in reality. So, if you are applying PCA at the
    # lower level, i.e., GW data at one inspection time step, without needing future data, normalization
    # is okay. But if you do so at the higher level, i.e., all GW data at all time steps until the end
    # of life, normalization is not possible in reality. a

    #1. Finding normalized matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #2. Finding covariance matrix of X and X_scaled
    covX = np.cov(X, rowvar=False)
    covX_scaled = np.cov(X_scaled, rowvar=False)

    #3. Finding eigenvectors+values
    eigenvalues,  eigenvectors = np.linalg.eig(covX_scaled)

    #4. Sort eigenvectors+values in descending order. Saving the total variance for later
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    total_variance = np.sum(eigenvalues)

    #5. Creating the S and inv(S) matrices - a diagonal matrix containing sqrt(eigenvalues)
    S = np.diag(np.sqrt(eigenvalues))
    S_inv = np.linalg.inv(S)

    #6. Creating the V and V_T matrices - the eigenvector and tranpose eigenvector matrices
    V = np.transpose(eigenvectors)
    V_T = eigenvectors

    #7. Creating the U matrix - Multiplying X_scaled x V x S_inv. We multiply 2 matrices at a time
    U = np.matmul(X_scaled, V)
    U = np.matmul(U, S_inv)

    #8. Truncating the S,U,V and eigenvalue matrices to only include columns up to the number of principal components
    # V_truncated contains the principal components - the n eigenvectors with the highest eigenvalues (these contain the most variance)
    U_truncated = U[:, :n]
    S_truncated = S[:n, :n]
    V_truncated = V[:, :n]
    eigenvalues_truncated = eigenvalues[:n]

    #9. Finding the normalized reduced dataset - Multiplying U_truncated x S_truncated x V_truncated. We multiply 2 matrices at a time
    PC = np.matmul(U_truncated, S_truncated)
    PC = np.matmul(PC, np.transpose(V_truncated))

    #10. Converting the reduced dataset back to original proportions, as currently it is normalized
    X_transf = scaler.inverse_transform(PC)

    #11. Return reduced dataset, the eigenvalues, the eigenvectors, and the original total variance
    return X_transf, eigenvalues_truncated, V_truncated, total_variance

def truncator(X):
    n, m = X.shape  # n = rows, m = columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    covX_scaled = np.cov(X_scaled, rowvar=False)
    eigenvalues,  eigenvectors = np.linalg.eig(covX_scaled)

    idx = eigenvalues.argsort()[::-1]
    singular_values = np.sqrt(eigenvalues[idx])

    # y_m: median value of S matrix (ie. median value of singular_values list)
    y_m = np.median(singular_values)

    # Calculate omega_approx
    # if m << n, beta = m / n and if n << m, beta = n / m
    if m < n:
        beta = m / n
    else:
        beta = n / m
    omega_approx = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    # threshold (%) is defined as threshold = y_m * omega_approx
    threshold = y_m * omega_approx

    # check if eigenvalue is more than threshold
    i = 0
    more_statement = 0
    while more_statement != 1:
        if singular_values[i] > threshold:
            more_statement = 0
        else:
            more_statement = 1
        i += 1
    r = i-1
    print("Truncation Value (ie. # of components kept) = {}" .format(r))
    return r

# if truncation function to be used: use n = truncator(final_breast_data)
#X_new, eigenvalues, eigenvectors, total_variance = PCA(final_breast_data, n=10)

#explained_variance = np.sum(eigenvalues / total_variance)*100
#print("Total explained variance:", explained_variance, "%")