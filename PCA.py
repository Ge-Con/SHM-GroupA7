import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target
labels = np.reshape(breast_labels, (569,1))

final_breast_data = np.concatenate([breast_data, labels], axis=1)
breast_dataset = pd.DataFrame(final_breast_data)
print(final_breast_data)

def PCA(X, n):

    #1. Finding normalized matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #2. Finding covariance matrix of X and X_scaled
    covX = np.cov(X, rowvar=False)
    covX_scaled = np.cov(X_scaled, rowvar=False)

    #3. Finding eigenvectors+values
    eigenvalues,  eigenvectors = np.linalg.eig(covX_scaled)

    #4. Sort eigenvectors+values in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    #5. Creating the S and inv(S) matrices - a diagonal matrix containing sqrt(eigenvalues)
    S = np.diag(np.sqrt(eigenvalues))
    S_inv = np.linalg.inv(S)

    #6. Creating the V and V_T matrices - the eigenvector and tranpose eigenvector matrices
    V = np.transpose(eigenvectors)
    V_T = eigenvectors

    #7. Creating the U matrix - Multiplying X_scaled x V x S_inv. We multiply 2 matrices at a time
    U = np.matmul(X_scaled, V)
    U = np.matmul(U, S_inv)

    #8. Truncating the S,U,V matrices to only include columns up to the number of principal components
    # V_truncated contains the principal components - the n eigenvectors with the highest eigenvalues (these contain the most variance)
    U_truncated = U[:, :n]
    S_truncated = S[:n, :n]
    V_truncated = V[:, :n]

    #9. Finding the normalized reduced dataset - Multiplying U_truncated x S_truncated x V_truncated. We multiply 2 matrices at a time
    PC = np.matmul(U_truncated, S_truncated)
    PC = np.matmul(PC, np.transpose(V_truncated))

    #10. Converting the reduced dataset back to original proportions, as currently it is normalized
    X_transf = scaler.inverse_transform(PC)

    #11. Return reduced dataset, the eigenvalues, the eigenvectors and the original covariance matrix
    return X_transf,S_truncated, V_truncated, covX

X_new, eigenvalues, eigenvectors, covX = PCA(final_breast_data, n=4)

original_var = np.trace(covX)
new_var = np.trace(np.cov(X_new, rowvar=False))
var_explained = (new_var / original_var)*100

print("Total explained variation:", var_explained, "%")

print()