from numpy.typing import NDArray
import numpy as np

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
