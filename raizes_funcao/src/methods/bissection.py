import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable


class Bissection:
    def __init__(
        self, xl: np.float64, xu: np.float64, tol: np.float64, max_iter: int
    ) -> None:
        self.__xl: np.float64 = xl
        self.__xu: np.float64 = xu
        self.__tol: np.float64 = tol
        self.__max_iter: int = max_iter

    def execute_method(self, func: Union[str, Callable]) -> np.float64:
        iter: int = 0
        relative_error: np.float64 = np.float64(100.0)
        x: np.float64 = self.__xl

        fl: np.float64 = evaluate_one_variable(func, self.__xl)

        while (relative_error > self.__tol) and (iter < self.__max_iter):
            x_old: np.float64 = x
            x = (self.__xl + self.__xu) / 2

            fx: np.float64 = evaluate_one_variable(func, x)

            iter = iter + 1

            if x != np.float64(0.0):
                relative_error = abs((x - x_old) / x) * 100

            if (fl * fx) < 0:
                self.__xu = x
            else:
                self.__xl = x
                fl = fx

        return x

    @property
    def xl(self) -> np.float64:
        return self.__xl

    @property
    def xu(self) -> np.float64:
        return self.__xu

    @property
    def max_iter(self) -> int:
        return self.__max_iter

    @xl.setter
    def xl(self, xl: np.float64) -> None:
        self.__xl = xl

    @xu.setter
    def xu(self, xu: np.float64) -> None:
        self.__xu = xu

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self.__max_iter = max_iter
