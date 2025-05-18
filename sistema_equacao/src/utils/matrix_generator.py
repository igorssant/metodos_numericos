import numpy as np
from numpy.typing import NDArray


def generate_hilbert_matrix(n: int) -> tuple[NDArray, NDArray]:
    """
    Generate an n x n Hilbert matrix.

    A Hilbert matrix is a square matrix with elements defined as:
    H(i, j) = 1 / (i + j - 1)

    Parameters:
    n (int): The size of the matrix.

    Returns:
    list: An n x n Hilbert matrix.
    """

    A: NDArray = np.zeros((n, n), dtype=np.float64)
    B: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
            B[i] += A[i, j]

    return A, B
