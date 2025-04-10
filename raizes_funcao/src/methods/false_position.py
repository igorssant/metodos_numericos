import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable


class FalsePosition:
    def __init__(
        self, xl: np.float64, xu: np.float64, tol: np.float64, max_iter: int
    ) -> None:
        self.__xl: np.float64 = xl
        self.__xu: np.float64 = xu
        self.__tol: np.float64 = tol
        self.__max_iter: int = max_iter

    def execute_method(self, func: Union[str, Callable]) -> np.float64:
        x: np.float64 = np.float64(0.0)
        iter: int = 0

        relative_error: np.float64 = np.float64(100.0)

        fl: np.float64 = evaluate_one_variable(func, self.__xl)
        fu: np.float64 = evaluate_one_variable(func, self.__xu)

        if abs(fl) < abs(fu):
            x = self.__xl
        else:
            x = self.__xu

        while (relative_error > self.__tol) and (iter < self.__max_iter):
            x_old: np.float64 = x
            iter = iter + 1
            x = self.__xu + (fu * (self.__xl - self.__xu)) / (fu - fl)

            fX: np.float64 = evaluate_one_variable(func, x)

            if x != np.float64(0.0):
                relative_error = abs((x - x_old) / x) * 100

            if abs(fl) < abs(fu):
                self.__xl = x
                fl = fX
            else:
                self.__xu = x
                fu = fX

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
