import numpy as np 
from enum import Enum

from synthetic_functions import RosenbrockFunction, LeastSquares, LevyFunction, QuingFunction, GriewankFunction, AckleyFunction, StyblinksiTangFunction


class ExperimentStatus(Enum):
    COMPLETED = 0
    NANFOUND = 1
    SKIP = 2



def get_objective_function(fun_name, d, seed):
    if fun_name == "Rosenbrock":
        return RosenbrockFunction(d=d, seed=seed)
    elif fun_name == "Griewank":
        return GriewankFunction(d=d, seed=seed)
    elif fun_name == "Ackley":
        return AckleyFunction(d=d, seed=seed)
    elif fun_name == "StyblinkskiTang":
        return StyblinksiTangFunction(d=d, seed=seed)
    elif fun_name == "Levy":
        return LevyFunction(d=d, seed=seed)
    raise ValueError(f"Function {fun_name} not recognized")