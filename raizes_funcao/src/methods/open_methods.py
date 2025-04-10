import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable


def fixed_point(
    func: Union[str, Callable], x0: np.float64, tol: np.float64, max_iter: int
) -> np.float64:
    """
    Método do Ponto Fixo para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        np.float64: A raiz encontrada.
    """

    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 = x0

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 = x
        iter = iter + 1

        x = evaluate_one_variable(func, x_old)

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

    return x


def newton_raphson(
    func: Union[str, Callable],
    derivative: Union[str, Callable],
    x0: np.float64,
    tol: np.float64,
    max_iter: int,
) -> np.float64:
    """
    Método de Newton-Raphson para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        derivative (Union[str, Callable]): A derivada da função.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        np.float64: A raiz encontrada.
    """

    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 = x0

    while (relative_error > tol) and (iter < max_iter):
        x0 = x
        iter = iter + 1
        
        x = x0 - (
            evaluate_one_variable(func, x0) / evaluate_one_variable(derivative, x0)
        )

        x = x0 - (
            evaluate_one_variable(func, x0) / evaluate_one_variable(derivative, x0)
        )

        if x != np.float64(0.0):
            relative_error = abs((x - x0) / x) * 100

    return x


def secant(
    func: Union[str, Callable],
    x0: np.float64,
    x1: np.float64,
    tol: np.float64,
    max_iter: int,
) -> np.float64:
    """
    Método da Secante para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        x1 (np.float64): O segundo ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        np.float64: A raiz encontrada.
    """

    relative_error: np.float64 = np.float64(100.0)
    iter: int = 0
    x: np.float64 = np.float64(0.0)

    while (relative_error > tol) and (iter < max_iter):
        iter = iter + 1

        f0: np.float64 = evaluate_one_variable(func, x0)
        f1: np.float64 = evaluate_one_variable(func, x1)

        x: np.float64 = x1 - f1 * ((x0 - x1) / (f0 - f1))

        if x != np.float64(0.0):
            relative_error = abs((x - x1) / x) * 100

        x0 = x1
        x1 = x

    return x
