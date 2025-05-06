from numpy.typing import NDArray
import numpy as np

def __calculate_error(xi:NDArray, x0:NDArray) -> np.float64:
    error_array:NDArray = np.zeros((xi.shape[0], 1), dtype=np.float64)
    
    for i in range(xi.shape[0]):
        error_array[i] = abs((xi[i] - x0[i]) / xi[i])
        
    return error_array.max()

def jacobi(augmented_matrix:NDArray, tol:np.float64, max_iter:int) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    initial_guess:NDArray = np.zeros(n_rows, dtype=np.float64)
    variable_values:NDArray = np.zeros(n_rows, dtype=np.float64)
    iter:int = 0
    
    for i in range(n_rows):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento diagonal nulo encontrado.")
        
        initial_guess[i] = augmented_matrix[i, n_cols - 1] / augmented_matrix[i, i]
        
    while(iter < max_iter):
        for i in range(n_rows):
            summ:np.float64 = np.float64(0.0)
            
            for j in range(i):
                summ += augmented_matrix[i, j] * initial_guess[j]
                
            for j in range(i + 1, n_cols - 2):
                summ += augmented_matrix[i, j] * initial_guess[j]
            
            variable_values[i] = (augmented_matrix[i, n_cols - 1] - summ) / augmented_matrix[i, i]

        if __calculate_error(variable_values, initial_guess) < tol:
            return variable_values
        
        iter += 1
            
    raise RuntimeError("O método de Jacobi não convergiu após ", max_iter, " iterações.")

def gauss_seidel(augmented_matrix:NDArray, tol:np.float64, max_iter:int) -> NDArray:
    pass

def relaxing(augmented_matrix:NDArray,
             tol:np.float64,
             max_iter:int,
             relax_factor:np.float64 = np.float64(1.0)) -> NDArray:
    pass
