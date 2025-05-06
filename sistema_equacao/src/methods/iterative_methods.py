from numpy.typing import NDArray
import numpy as np

def __calculate_error(xi:NDArray, x0:NDArray) -> np.float64:
    error_array:NDArray = np.abs((xi - x0) / xi)
    
    return np.max(error_array)

def jacobi(augmented_matrix:NDArray, tol:np.float64, max_iter:int) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    initial_guess:NDArray = np.zeros(n_rows, dtype=np.float64)
    variable_values:NDArray = np.zeros(n_rows, dtype=np.float64)
    iter:int = 0
    
    for i in range(n_rows):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento nulo encontrado  na diagonal.")
        
        initial_guess[i] = augmented_matrix[i, n_cols - 1] / augmented_matrix[i, i]
        
    while(iter < max_iter):
        for i in range(n_rows):
            summ:np.float64 = np.float64(0.0)
            
            for j in range(i):
                summ += augmented_matrix[i, j] * initial_guess[j]
                
            for j in range(i + 1, n_cols - 1):
                summ += augmented_matrix[i, j] * initial_guess[j]
            
            variable_values[i] = (augmented_matrix[i, n_cols - 1] - summ) / augmented_matrix[i, i]

        if __calculate_error(variable_values, initial_guess) < tol:
            return variable_values
        
        iter += 1
        initial_guess = np.copy(variable_values)
            
    raise RuntimeError("O método de Jacobi não convergiu após ", max_iter, " iterações.")

def gauss_seidel(augmented_matrix:NDArray, tol:np.float64, max_iter:int) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    variable_values:NDArray = np.zeros(n_rows, dtype=np.float64)
    iter:int = 0
    
    for i in range(n_rows):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento nulo encontrado na diagonal.")
        
        variable_values[i] = augmented_matrix[i, n_cols - 1] / augmented_matrix[i, i]
        
    while(iter < max_iter):
        initial_guess:NDArray = np.copy(variable_values)

        for i in range(n_rows):
            summ:np.float64 = np.float64(0.0)
            
            for j in range(i):
                summ += augmented_matrix[i, j] * variable_values[j]
                
            for j in range(i + 1, n_cols - 1):
                summ += augmented_matrix[i, j] * variable_values[j]
            
            variable_values[i] = (augmented_matrix[i, n_cols - 1] - summ) / augmented_matrix[i, i]

        if __calculate_error(variable_values, initial_guess) < tol:
            return variable_values
        
        iter += 1
            
    raise RuntimeError("O método de Gauss-Seidel não convergiu após ", max_iter, " iterações.")

def relaxing(augmented_matrix:NDArray, tol:np.float64, max_iter:int, relax_factor:np.float64 = np.float64(1.0)) -> NDArray:
    n_rows:int = augmented_matrix.shape[0]
    n_cols:int = augmented_matrix.shape[1]
    variable_values:NDArray = np.zeros(n_rows, dtype=np.float64)
    iter:int = 0
    
    for i in range(n_rows):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento nulo encontrado na diagonal.")
        
        variable_values[i] = augmented_matrix[i, n_cols - 1] / augmented_matrix[i, i]
        
    while iter < max_iter:
        previous_values:NDArray = np.copy(variable_values)

        for i in range(n_rows):
            summ: np.float64 = np.float64(0.0)
            gauss_seidel_result:np.float64 = np.float64(0.0)
            
            for j in range(i):
                summ += augmented_matrix[i, j] * variable_values[j]
                
            for j in range(i + 1, n_cols - 1):
                summ += augmented_matrix[i, j] * variable_values[j]
                    
            gauss_seidel_result = (augmented_matrix[i, n_cols - 1] - summ) / augmented_matrix[i, i]
            variable_values[i] = (1 - relax_factor) * previous_values[i] + relax_factor * gauss_seidel_result

        if __calculate_error(variable_values, previous_values) < tol:
            return variable_values

        iter += 1

    raise RuntimeError(f"O método do Relaxamento não convergiu após {max_iter} iterações.")
