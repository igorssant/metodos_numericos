from numpy.typing import NDArray
import numpy as np
from typing import Callable


def partial_pivoting(a: NDArray, b: NDArray) -> None:
    """Performs partial pivoting on the given matrix a and vector b."""

    n: int = a.shape[0]

    for k in range(n):
        # Find the index of the maximum element in the current column
        max_index = np.argmax(np.abs(a[k:, k])) + k

        # Swap the rows in a
        a[[k, max_index], :] = a[[max_index, k], :]

        # Swap the corresponding elements in b
        b[k], b[max_index] = b[max_index], b[k]


def partial_pivoting_decorator(func: Callable[[NDArray, NDArray], NDArray]):
    def wrapper(a, b, *args, **kwargs):
        # Aplicando o pivotamento parcial
        partial_pivoting(a, b)

        # Chamando a função original
        return func(a, b, *args, **kwargs)

    return wrapper


def eliminate(a: NDArray, b: NDArray) -> None:
    """Perfoms Gaussian elimination on the given matrix a and vector b."""

    n: int = a.shape[0]

    for k in range(n):
        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]

            for j in range(k + 1, n):
                a[i, j] = a[i, j] - factor * a[k, j]

            b[i] = b[i] - factor * b[k]


def back_substitution(a: NDArray, b: NDArray) -> NDArray:
    """Performs back substitution on the given matrix a and vector b."""

    n: int = a.shape[0]
    x: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, i + 1 :], x[i + 1 :])) / a[i, i]

    return x


def naive_gauss(a: NDArray, b: NDArray) -> NDArray:
    """Performs Gaussian elimination on the given augmented matrix.
    The augmented matrix is expected to be in the form [A|b], where A is the coefficient matrix and b is the constant vector.
    The function returns the solution vector x.
    """
    n: int = a.shape[0]

    if n != a.shape[1] or n != b.shape[0]:
        raise ValueError("The matrix must be square.")

    eliminate(a, b)
    return back_substitution(a, b)


@partial_pivoting_decorator
def gauss_partial_pivoting(a: NDArray, b: NDArray) -> NDArray:
    """Performs Gaussian elimination with partial pivoting on the given matrix a and vector b.
    The function returns the solution vector x.
    """

    return naive_gauss(a, b)

