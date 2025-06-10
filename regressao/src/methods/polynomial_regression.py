from numpy.typing import NDArray
import numpy as np

def create_augmented_matrix(x:NDArray, y:NDArray, poly_order:int) -> NDArray:
    data_size:int = len(x)
    
    if data_size < poly_order + 1:
        raise ArithmeticError(f"""
                               Erro: O número de pontos de dados deve ser maior ou igual a ordem do polinômio + 1.
                               Quantidade de dados = {data_size}.
                               Ordem + 1 = {poly_order + 1}.
                               Regressão impossível.
                               """)
    
    augmented_matrix :NDArray = np.zeros((poly_order + 1,
                                          poly_order + 2),
                                         dtype=np.float64)

    for i in range(1 + poly_order):
        for j in range(1 + i):
            k :int = i + j
            summ :np.float64 = np.float64(0.0)

            for l in range(j):
                summ += x[l]**2

            for l in range(j + 1, data_size):
                summ += x[l]**2

            augmented_matrix[i, j] = summ
            summ = np.float64(0.0)

        for j in range(data_size):
            summ += y[j] * x[j]**i
        
        augmented_matrix[i, poly_order + 1] = summ
        
    return augmented_matrix

def retrieve_coef(A: NDArray, b:NDArray) -> NDArray:
    return np.linalg.solve(A, b)
