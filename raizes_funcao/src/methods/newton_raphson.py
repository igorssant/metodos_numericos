import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable

class NewtonRaphson:
    def __init__(self, x0:np.float64, tol:np.float64, max_iter:int) -> None:
        self.__x0:np.float64 = x0
        self.__tol:np.float64 = tol
        self.__max_iter:int = max_iter

    def execute_method(self, func:Union[str, Callable], derivative:Union[str, Callable]) -> np.float64:
        iter:int = 0
        relative_error:np.float64 = np.float64(100.0)
        x:np.float64 = self.__x0

        while (relative_error > self.__tol) and (iter < self.__max_iter):
            self.__x0  = x
            iter = iter + 1
<<<<<<< HEAD
            x = self.__x0 - (evaluate_one_variable(func, self.__x0) /
                            evaluate_one_variable(derivative, self.__x0))
=======
>>>>>>> 5a3cb38 (feat: extend _parser para reconhecer expresões numpy)
            
            x = self.__x0 - (evaluate_one_variable(func, self.__x0) /
                            evaluate_one_variable(derivative, self.__x0))

            if x != np.float64(0.0):
                relative_error = abs((x - self.__x0) / x) * 100

        return x

    @property
    def x0(self) -> np.float64:
        return self.__x0

    @property
    def max_iter(self) -> int:
        return self.__max_iter

    @x0.setter
    def x0(self, x0: np.float64) -> None:
        self.__x0 = x0

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self.__max_iter = max_iter
