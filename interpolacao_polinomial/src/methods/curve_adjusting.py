import sys
import os

from numpy.typing import NDArray
import numpy as np

# Adicionar o diretório 'src' (o diretório raiz do projeto de módulos) ao Python path
# Agora ele vai um nível acima do diretório 'methods' para chegar em 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gauss_seidel import (
    gauss_seidel, get_augmented_matrix
)

def natural_splines(X: NDArray[np.float64],
                    Y: NDArray[np.float64],):

    # Inicializar os coeficientes dos splines
    a, b, c, d = np.zeros(len(X)-1), np.zeros(len(X)-1), np.zeros(len(X)-1), np.zeros(len(X)-1)

    # Número de splines
    n = X.shape[0] - 1

    # a é o valor de Y nos pontos X
    a = Y[:-1]

    # h é a diferença entre os pontos X
    # h[i] = X[i+1] - X[i]
    h = np.diff(X)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    # Construir o sistema de equações para c
    for j in range(1, n):
        A[j, j - 1] = h[j - 1]
        A[j, j] = 2 * (h[j - 1] + h[j])
        A[j, j + 1] = h[j]

        B[j] = (3 / h[j]) * (a[j] - a[j - 1]) - (3 / h[j - 1]) * (a[j - 1] - a[j - 2])

    print("N : ", n)
    print("Vetor h\n", h)
    print("Matriz A\n", A)
    print("Matriz B\n", B)
    augmented_matrix = get_augmented_matrix(A, B)

    c = gauss_seidel(augmented_matrix, 1e-8, 500)

    for j in range(1, n - 1):
        b[j] = (1 / h[j - 1]) * (a[j] - a[j - 1]) - (h[j - 1] / 3) * (2 * c[j - 1] + c[j])
        d[j] = (1 / (3 * h[j])) * (c[j + 1] - c[j])

    return a, b, c, d


if __name__  == "__main__":
    # Exemplo de uso
    X = np.array([0, 1, 2, 3], dtype=np.float64)
    Y = np.array([1, 2, 0, 3], dtype=np.float64)

    a, b, c, d = natural_splines(X, Y)

    print("Coeficientes dos splines:")
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
