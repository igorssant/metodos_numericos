from numpy.typing import NDArray
import numpy as np

def determinate_elements(X:NDArray, b:NDArray) -> NDArray:
    data_size :int = b.shape[0]
    num_variables :int = X.shape[1] if X.ndim > 1 else 1
    poly_order :int = num_variables
    aux_matrix :NDArray = np.hstack((np.ones((data_size, 1)), X))
    augmented_matrix :NDArray = np.zeros((poly_order + 1,
                                          poly_order + 2),
                                         dtype=np.float64)

    for i in range(poly_order + 1):
        for j in range(i + 1):
            summ :np.float64 = np.float64(0.0)

            for l in range(data_size):
                summ += aux_matrix[l, i] * aux_matrix[l, j]

            augmented_matrix[i, j] = summ

            if i != j:
                augmented_matrix[j, i] = summ

        summ = np.float64(0.0)

        for l in range(data_size):
            summ += b[l] * aux_matrix[l, i]

        augmented_matrix[i, poly_order + 1] = summ
        
    return augmented_matrix

def retrieve_multiple_coef(A:NDArray, b:NDArray) -> NDArray:
    return np.linalg.solve(A, b)
