from numpy.typing import NDArray
from typing import Tuple
import numpy as np

def gaussian_elimination(augmented_matrix:NDArray) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    solution_array:NDArray = np.zeros(n_cols - 1, dtype=np.float64)
    
    for i in range(n_rows):
        if np.isclose(augmented_matrix[i, i], np.float64(0.0)):
            raise ArithmeticError("O método falhou!")
        
        for j in range(i + 1, n_rows):
            scale_factor:np.float64 = augmented_matrix[j, i] / augmented_matrix[i, i]
            
            for k in range(i, n_cols):
                augmented_matrix[j, k] = augmented_matrix[j, k] - scale_factor * augmented_matrix[i, k]
            
    if np.isclose(augmented_matrix[n_rows - 1, n_rows - 1], np.float64(0.0)):
        raise ArithmeticError("Não existe solução única!")
    
    solution_array[n_cols - 2] = augmented_matrix[n_rows - 1, n_cols - 1] / augmented_matrix[n_rows - 1, n_cols - 2]
    
    for i in range(n_cols - 3, -1, -1):
        summ:np.float64 = np.float64(0.0)
        
        for j in range(i + 1, n_cols - 1):
            summ += augmented_matrix[i, j] * solution_array[j]
        
        solution_array[i] = (augmented_matrix[i, n_cols - 1] - summ) / augmented_matrix[i, i]
    
    return solution_array

def gaussian_elimination_partial_pivoting(augmented_matrix:NDArray) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    solution_array:NDArray = np.zeros(n_cols - 1, dtype=np.float64)
    nlin:list[int] = list(range(n_rows))
        
    for i in range(n_cols - 1):
        pivot_row:int = i
        
        for k in range(i + 1, n_rows):
            if abs(augmented_matrix[nlin[pivot_row], i]) < abs(augmented_matrix[nlin[k], i]):
                pivot_row = k
                
        if np.isclose(augmented_matrix[nlin[pivot_row], i], np.float64(0.0)):
            raise ArithmeticError("Não existe solução única!")
        
        if nlin[i] != nlin[pivot_row]:
            nlin[i], nlin[pivot_row] = nlin[pivot_row], nlin[i]
            
        for j in range(i + 1, n_rows):
            scale_factor:np.float64 = augmented_matrix[nlin[j], i] / augmented_matrix[nlin[i], i]
            
            for k in range(i, n_cols):
                augmented_matrix[nlin[j], k] = augmented_matrix[nlin[j], k] - scale_factor * augmented_matrix[nlin[i], k]

    if np.isclose(augmented_matrix[nlin[n_rows - 1], n_cols - 1], np.float64(0.0)):
        raise ArithmeticError("Não existe solução única!")
    
    solution_array[n_cols - 2] = augmented_matrix[nlin[n_rows - 1], n_cols - 1] / augmented_matrix[nlin[n_rows - 1], n_cols - 2]
    
    for i in range(n_cols - 3, -1, -1):
        summ:np.float64 = np.float64(0.0)
        
        for j in range(i + 1, n_cols - 1):
            summ += augmented_matrix[nlin[i], j] * solution_array[j]
            
        solution_array[i] = (augmented_matrix[nlin[i], n_cols - 1] - summ) / augmented_matrix[nlin[i], i]
    
    return solution_array

def LU_factoring(square_matrix:NDArray) -> Tuple[NDArray, NDArray]:
    length:int = square_matrix.shape[0]
    L:NDArray = np.zeros((length, length), dtype=np.float64)
    U:NDArray = np.zeros((length, length), dtype=np.float64)

    for i in range(length):
        for j in range(i, length):
            summ:np.float64 = np.sum(L[i, :i] * U[:i, j])
            U[i, j] = square_matrix[i, j] - summ

        for j in range(i, length):
            if U[i, i] == 0:
                raise ArithmeticError("Fatoração LU não é possível!\nMatriz é singular.")
            
            summ:np.float64 = np.sum(L[j, :i] * U[:i, i])
            L[j, i] = (square_matrix[j, i] - summ) / U[i, i]

    return (L, U)
