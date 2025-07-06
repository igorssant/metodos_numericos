import sys
import os
from typing import Callable, Tuple
from numpy.typing import NDArray
import numpy as np

# Adicionar o diretório 'src' (o diretório raiz do projeto de módulos) ao Python path
# Agora ele vai um nível acima do diretório 'methods' para chegar em 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gauss_seidel import gauss_seidel, get_augmented_matrix
from utils.parser import evaluate_one_variable, get_derivative


def natural_splines(
    X: NDArray[np.float64], Y: NDArray[np.float64]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    # Número de splines
    n = X.shape[0] - 1

    # Inicializar os coeficientes dos splines
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    # a é o valor de Y nos pontos X
    a = Y[:-1]

    # h é a diferença entre os pontos X
    # h[i] = X[i+1] - X[i]
    h = np.diff(X)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    # Construir o sistema de equações para c nos pontos internos
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        B[i] = (3 / h[i]) * (Y[i + 1] - Y[i]) - (3 / h[i - 1]) * (Y[i] - Y[i - 1])

    augmented_matrix = get_augmented_matrix(A, B)

    # Resolver o sistema para encontrar os coeficientes c
    c = gauss_seidel(augmented_matrix, 1e-6, 200)

    # Calcular os coeficientes b
    for i in range(n):
        b[i] = (1 / h[i]) * (Y[i + 1] - Y[i]) - (h[i] / 3) * (2 * c[i] + c[i + 1])

    # Calcular os coeficientes d
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Remover o último elemento, pois não é necessário
    c = c[:-1]

    return (a, b, c, d)


def fixed_splines(
    X: NDArray[np.float64], Y: NDArray[np.float64], dx_0: np.float64, dx_n: np.float64
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    n = X.shape[0] - 1
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    a = Y[:-1]
    h = np.diff(X)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    # Condição de contorno inicial: S'(x₀) = dx_0
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    B[0] = 3 * (Y[1] - Y[0]) / h[0] - 3 * dx_0

    # Condição de contorno final: S'(xₙ) = dx_n
    A[n, n - 1] = h[n - 1]
    A[n, n] = 2 * h[n - 1]
    B[n] = 3 * dx_n - 3 * (Y[n] - Y[n - 1]) / h[n - 1]

    # Equações internas (continuidade da segunda derivada)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * (Y[i + 1] - Y[i]) / h[i] - 3 * (Y[i] - Y[i - 1]) / h[i - 1]

    augmented_matrix = get_augmented_matrix(A, B)
    c_full = gauss_seidel(augmented_matrix, 1e-6, 200)

    # Calcular coeficientes b e d
    for i in range(n):
        b[i] = (Y[i + 1] - Y[i]) / h[i] - h[i] * (2 * c_full[i] + c_full[i + 1]) / 3
        d[i] = (c_full[i + 1] - c_full[i]) / (3 * h[i])

    c = c_full[:-1]  # Remover último elemento
    return (a, b, c, d)


def cubic_splines(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    dx_0: np.float64 | None = None,
    dx_n: np.float64 | None = None,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    if dx_0 is not None and dx_n is not None:
        return fixed_splines(X, Y, dx_0=dx_0, dx_n=dx_n)
    else:
        return natural_splines(X, Y)


def evaluate_spline(
    x: float,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    x_points: NDArray[np.float64],
) -> float:
    """
    Avalia o polinômio spline cúbico em um ponto x dado os coeficientes a, b, c, d e os pontos x_points.
    """
    n = len(x_points) - 1
    for i in range(n):
        if x_points[i] <= x <= x_points[i + 1]:
            return (
                a[i]
                + b[i] * (x - x_points[i])
                + c[i] * (x - x_points[i]) ** 2
                + d[i] * (x - x_points[i]) ** 3
            )
    raise ValueError("x está fora do intervalo dos pontos dados.")


def get_spline_func_str(
    x: float,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    x_points: NDArray[np.float64],
) -> str:
    """
    Retorna uma função que avalia o polinômio spline cúbico dado os coeficientes a, b, c, d e os pontos x_points.
    """
    n = len(x_points) - 1
    for i in range(n):
        if x_points[i] <= x <= x_points[i + 1]:
            return f"({a[i]} + {b[i]} * (x - {x_points[i]}) + {c[i]} * (x - {x_points[i]})**2 + {d[i]} * (x - {x_points[i]})**3)"

    raise ValueError("x está fora do intervalo dos pontos dados.")


if __name__ == "__main__":
    func :str = "cos(pi * x)"
    true_derivs_5 :NDArray[np.float64] = np.array([
        [0.0,
         np.pi * np.sqrt(2.0) / 2.0,
         -np.pi,
         -np.pi * np.sqrt(2.0) / 2.0,
         0.0],
        [-np.pi**2,
         np.pi**2 * np.sqrt(2.0) / 2.0,
         0.0,
         np.pi**2 * np.sqrt(2.0) / 2.0,
         np.pi**2]
        ], dtype=np.float64)
    true_derivs_7 :NDArray[np.float64] = np.array([
        [0.0,
         -np.pi * np.sqrt(2.0 - np.sqrt(2.0)) / 2.0,
         -np.pi * np.sqrt(2.0) / 2.0,
         -np.pi * np.sqrt(2.0 + np.sqrt(2.0)) / 2.0,
         -np.pi,
         -np.pi * np.sqrt(2.0 + np.sqrt(2.0)) / 2.0,
         -np.pi * np.sqrt(2.0) / 2.0,
         -np.pi * np.sqrt(2.0 - np.sqrt(2.0)) / 2.0,
         0.0],
        [-np.pi**2,
         -np.pi**2 * np.sqrt(2.0 + np.sqrt(2.0)) / 2.0,
         -np.pi**2 * np.sqrt(2.0) / 2.0,
         -np.pi**2 * np.sqrt(2.0 - np.sqrt(2.0)) / 2.0,
         0.0,
         -np.pi**2 * (-np.sqrt(2.0 - np.sqrt(2.0))) / 2.0,
         -np.pi**2 * (-np.sqrt(2.0)) / 2.0,
         -np.pi**2 * (-np.sqrt(2.0 + np.sqrt(2.0))) / 2.0,
         np.pi**2]
        ], dtype=np.float64)
    
    print("Caso NATURAL:")
    
    x :NDArray[np.float64] = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    y :NDArray[np.float64] = np.array([evaluate_one_variable(func, el) for el in x], dtype=np.float64)
    derivatives_list :NDArray[np.float64] = np.zeros((2, x.shape[0]),
                                                     dtype=np.float64)
    coefs :Tuple[np.float64,
                 np.float64,
                 np.float64,
                 np.float64] = cubic_splines(x,y)
    
    for i in range(x.shape[0]):
        spline_func :str = get_spline_func_str(x[i], *coefs, x)
        derivative_func :Callable[[float], np.float64] = get_derivative(spline_func, degree=1)
        derivative_func_2 :Callable[[float], np.float64] = get_derivative(spline_func, degree=2)
        derivatives_list[0, i] = derivative_func(x[i])
        derivatives_list[1, i] = derivative_func_2(x[i])
        
    print("Derivadas de S_5:\n", derivatives_list[:, 0])
    print("Segunda derivadas de S_5:\n", derivatives_list[:, 1])
    print("Erro primeira derivada:\n", abs(true_derivs_5[0, :] - derivatives_list[0, :]))
    print("Erro segunda derivada:\n", abs(true_derivs_5[1, :] - derivatives_list[1, :]))
    
    x = np.array([0.0, 0.125, 0.250, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float64)
    y = np.array([evaluate_one_variable(func, el) for el in x], dtype=np.float64)
    derivatives_list = np.zeros((2, x.shape[0]), dtype=np.float64)
    coefs = cubic_splines(x,y)
    
    for i in range(x.shape[0]):
        spline_func :str = get_spline_func_str(x[i], *coefs, x)
        derivative_func :Callable[[float], np.float64] = get_derivative(spline_func, degree=1)
        derivative_func_2 :Callable[[float], np.float64] = get_derivative(spline_func, degree=2)
        derivatives_list[0, i] = derivative_func(x[i])
        derivatives_list[1, i] = derivative_func_2(x[i])
        
    print("Derivadas de S_9: ", derivatives_list[:, 0])
    print("Segunda derivadas de S_9: ", derivatives_list[:, 1])
    print("Erro primeira derivada:\n", abs(true_derivs_7[0, :] - derivatives_list[0, :]))
    print("Erro segunda derivada:\n", abs(true_derivs_7[1, :] - derivatives_list[1, :]))
    print("\n=================================================")
    print("Caso FIXADO:")
    
    deriv :Callable[[float], np.float64] = get_derivative(func)
    
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    y = np.array([evaluate_one_variable(func, el) for el in x], dtype=np.float64)
    derivatives_list = np.zeros((2, x.shape[0]), dtype=np.float64)
    coefs = cubic_splines(x,
                          y,
                          dx_0=deriv(x[0]),
                          dx_n=deriv(x[-1]))
    
    for i in range(x.shape[0]):
        spline_func :str = get_spline_func_str(x[i], *coefs, x)
        derivative_func :Callable[[float], np.float64] = get_derivative(spline_func, degree=1)
        derivative_func_2 :Callable[[float], np.float64] = get_derivative(spline_func, degree=2)
        derivatives_list[0, i] = derivative_func(x[i])
        derivatives_list[1, i] = derivative_func_2(x[i])
        
    print("Derivadas de S_5: ", derivatives_list[:, 0])
    print("Segunda derivadas de S_5: ", derivatives_list[:, 1])
    print("Erro primeira derivada:\n", abs(true_derivs_5[0, :] - derivatives_list[0, :]))
    print("Erro segunda derivada:\n", abs(true_derivs_5[1, :] - derivatives_list[1, :]))
    
    x = np.array([0.0, 0.125, 0.250, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float64)
    y = np.array([evaluate_one_variable(func, el) for el in x], dtype=np.float64)
    derivatives_list = np.zeros((2, x.shape[0]), dtype=np.float64)
    coefs = cubic_splines(x,
                          y,
                          dx_0=deriv(x[0]),
                          dx_n=deriv(x[-1]))
    
    for i in range(x.shape[0]):
        spline_func :str = get_spline_func_str(x[i], *coefs, x)
        derivative_func :Callable[[float], np.float64] = get_derivative(spline_func, degree=1)
        derivative_func_2 :Callable[[float], np.float64] = get_derivative(spline_func, degree=2)
        derivatives_list[0, i] = derivative_func(x[i])
        derivatives_list[1, i] = derivative_func_2(x[i])
        
    print("Derivadas de S_9: ", derivatives_list[:, 0])
    print("Segunda derivadas de S_9: ", derivatives_list[:, 1])
    print("Erro primeira derivada:\n", abs(true_derivs_7[0, :] - derivatives_list[0, :]))
    print("Erro segunda derivada:\n", abs(true_derivs_7[1, :] - derivatives_list[1, :]))
