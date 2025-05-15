from numpy.typing import NDArray
import numpy as np

def newton_interpolation(pointwise_matrix:NDArray, poly_order:int, x0:np.float64) -> tuple[NDArray, NDArray]:
    # return (yinter, abs_err)
    pass
    
def lagrange_interpolation(pointwise_matrix:NDArray, poly_order:int, x0:np.float64) -> NDArray:
    summ:np.float64 = np.float64(0.0)
    
    for k in range(poly_order + 1):
        prod:np.float64 = np.float64(1.0)
        i:int = 0
        
        while(i < k):
            prod *= (x0 - pointwise_matrix[0, i]) / (pointwise_matrix[0, k] - pointwise_matrix[0, i])
            i += 1
            
        i = k + 1
        
        while(i < poly_order + 1):
            prod *= (x0 - pointwise_matrix[0, i]) / (pointwise_matrix[0, k] - pointwise_matrix[0, i])
            i += 1
            
        summ += prod * pointwise_matrix[1, k]
    
    return summ
