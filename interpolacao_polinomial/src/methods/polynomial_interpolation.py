from numpy.typing import NDArray
import numpy as np

def newton_interpolation(pointwise_matrix:NDArray, xi:np.float64) -> tuple[NDArray, NDArray]:
    n:int = pointwise_matrix.shape[0]
    fdd:NDArray = np.zeros((n + 1, n + 1), dtype=np.float64)
    ea:np.float64 = np.zeros(n, dtype=np.float64)
    xterm:np.float64 = np.float64(1.0)
    yint:np.float64 = np.float64(0.0)

    fdd[:, 0] = pointwise_matrix[1, :]

    for j in range(1, n + 1):
        for i in range(n - j + 1):
            fdd[i, j] = (fdd[i + 1, j - 1] - fdd[i, j - 1]) / (pointwise_matrix[0, i + j] - pointwise_matrix[0, i])

    yint = fdd[0, 0]

    for order in range(1, n + 1):
        xterm = xterm * (xi - pointwise_matrix[0, order - 1])
        yint_new = yint + fdd[0, order] * xterm
        ea[order - 1] = np.abs(yint_new - yint)
        yint = yint_new

    return (yint, ea)
    
def lagrange_interpolation(pointwise_matrix:NDArray, poly_order:int, xi:np.float64) -> NDArray:
    summ:np.float64 = np.float64(0.0)
    
    for k in range(poly_order + 1):
        prod:np.float64 = np.float64(1.0)
        i:int = 0
        
        while(i < k):
            prod *= (xi - pointwise_matrix[0, i]) / (pointwise_matrix[0, k] - pointwise_matrix[0, i])
            i += 1
            
        i = k + 1
        
        while(i < poly_order + 1):
            prod *= (xi - pointwise_matrix[0, i]) / (pointwise_matrix[0, k] - pointwise_matrix[0, i])
            i += 1
            
        summ += prod * pointwise_matrix[1, k]
    
    return summ
