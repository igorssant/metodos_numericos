from numpy.typing import NDArray
import numpy as np

def create_augmented_matrix_optimized(x: NDArray, y: NDArray, poly_order: int) -> NDArray:
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
