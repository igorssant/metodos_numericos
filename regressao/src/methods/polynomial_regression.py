from numpy.typing import NDArray
import numpy as np

def create_augmented_matrix(x: NDArray, y: NDArray, poly_order: int) -> NDArray:
    data_size = len(x)
    if data_size < poly_order + 1:
        raise ArithmeticError("Erro: O número de pontos deve ser maior ou igual a ordem do polinômio + 1.")

    augmented_matrix = np.zeros((poly_order + 1, poly_order + 2), dtype=np.float64)

    for i in range(poly_order + 1):
        for j in range(i + 1): 
            sum_x_powers = np.sum(x**(i + j))
            augmented_matrix[i, j] = sum_x_powers

            if i != j:
                augmented_matrix[j, i] = sum_x_powers

        augmented_matrix[i, poly_order + 1] = np.sum(y * (x**i))

    return augmented_matrix

def retrieve_poly_coef(A: NDArray, b:NDArray) -> NDArray:
    return np.linalg.solve(A, b)

def __calculate_squared_sum(y_true :NDArray) -> np.float64:
    St :np.float64 = np.float64(0.0)
    mean :np.float64 = np.mean(y_true, dtype=np.float64)
    
    for i in range(y_true.shape[0]):
        St += (y_true[i] - mean)**2

    return St

def __calculate_squared_sum_error(y_estimate :NDArray, y_true :NDArray) -> np.float64:
    Sr :np.float64 = np.float64(0.0)
    
    for i in range(y_estimate.shape[0]):
        Sr += (y_true[i] - y_estimate[i])**2

    return Sr

def calculate_r2(y_estimate :NDArray, y_true :NDArray) -> np.float64:
    St :np.float64 = __calculate_squared_sum(y_true)
    Sr :np.float64 = __calculate_squared_sum_error(y_estimate, y_true)
    
    return (St - Sr) / St
