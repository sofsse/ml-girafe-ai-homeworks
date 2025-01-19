import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvector = np.random.random(size=(data.shape[0],)) # starting with random initialization
    for _ in range(num_steps):
        eigenvector = (data @ eigenvector) / np.linalg.norm(data @ eigenvector)

    eigenvalue = (eigenvector.T @ data @ eigenvector) / (eigenvector.T @ eigenvector)

    return float(eigenvalue), eigenvector