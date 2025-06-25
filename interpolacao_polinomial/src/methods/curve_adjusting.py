from numpy.typing import NDArray
import numpy as np

def natural_splines(X: NDArray[np.float64],
            Y: NDArray[np.float64],):

    # Inicializar os coeficientes dos splines
    a, b, c, d = np.zeros(len(X)-1), np.zeros(len(X)-1), np.zeros(len(X)-1), np.zeros(len(X)-1)

    # Número de splines
    n = len(X) - 1

    # a é o valor de Y nos pontos X
    a =  Y[:-1]

    # h é a diferença entre os pontos X
    # h[i] = X[i+1] - X[i]
    h = np.diff(X)

    c[0] = 0  # Condição de contorno natural
    c[-1] = 0  # Condição de contorno natural


    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    # Construir o sistema de equações para c
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A [i, i + 1] = h[i]

        B[i] = (3 / h[i]) * (a[i + 1] - a[i] - (3 / h[i - 1]))

    
