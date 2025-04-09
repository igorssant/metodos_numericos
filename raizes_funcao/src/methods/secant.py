import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable

class Secant:
    def __init__(self, x0:np.float64, x1:np.float64, tol:np.float64, max_iter:int) -> None:
        self.__x0:np.float64 = x0
        self.__x1:np.float64 = x1
        self.__tol:np.float64 = tol
        self.__max_iter:int = max_iter

    def executeMethod(self, func:Union[str, Callable]) -> np.float64:
        relative_error:np.float64 = np.float64(100.0)
        
        while (relative_error > self.__tol) and (iter < self.__max_iter):
            iter = iter + 1
            f0:np.float64 = evaluate_one_variable(self._function, self.__x0)
            f1:np.float64 = evaluate_one_variable(self._function, self.__x1)
            x:np.float64 = self.__x1 - f1 * ((self.__x0 - self.__x1) / (f0 - f1))
            
            if x != np.float64(0.0):
                relative_error = abs((x - self.__x1) / x) * 100
            
            self.__x0 = self.__x1
            self.__x1 = x
            
        return x

    @property
    def x0(self) -> np.float64:
        return self.__x0
    
    @property
    def x1(self) -> np.float64:
        return self.__x1

    @property
    def max_iter(self) -> int:
        return self.__max_iter

    @x0.setter
    def x0(self, x0: np.float64) -> None:
        self.__x0 = x0
        
    @x1.setter
    def x1(self, x1: np.float64) -> None:
        self.__x1 = x1

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self.__max_iter = max_iter