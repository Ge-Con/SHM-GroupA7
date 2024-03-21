    import pandas as pd
    import numpy as np
    from main import list_files
    import csv
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
        #2. Finding covariance matrix
        covX = np.cov(X_scaled, rowvar=False)
        #3. Finding eigenvectors+values
        eigenvalues,  eigenvectors = np.linalg.eig(covX)

        #4. Sort eigenvectors+values
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        print("------")
        S = np.diag(np.sqrt(eigenvalues))
        S_inv = np.linalg.inv(S)
        V = np.transpose(eigenvectors)
        V_T = eigenvectors
        U = np.matmul(X_scaled, V)
        U = np.matmul(U, S_inv) 
        #5. Transform original data
        A = np.matmul(U, S)
        A = np.matmul(A, V_T)
        U_t = U[:, :n]
        S_t = S[:n, :n]
        V_t = V[:, :n]
        PC = np.matmul(U_t, S_t)
        PC = np.matmul(PC, np.transpose(V_t))
        X_transf = scaler.inverse_transform(PC)
        return X_transf,S_t, V_t
    print(PCA(final_breast_data, 10)[0])
