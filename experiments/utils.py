import numpy as np 
from enum import Enum

from synthetic_functions import * #RosenbrockFunction, LeastSquares, RastriginFunction, LevyFunction, QuingFunction, GriewankFunction, AckleyFunction, StyblinksiTangFunction


class ExperimentStatus(Enum):
    COMPLETED = 0
    NANFOUND = 1
    SKIP = 2



def get_objective_function(fun_name, d, reg, seed):
    if fun_name == "Rosenbrock":
        return RosenbrockFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == "Griewank":
        return GriewankFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == "Ackley":
        return AckleyFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == "StyblinkskiTang":
        return StyblinksiTangFunction(d=d,regularization=reg, seed=seed)
    elif fun_name == "Quing":
        return QuingFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == "Rastrigin":
        return RastriginFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == "Levy":
        return LevyFunction(d=d, regularization=reg, seed=seed)
    elif fun_name == 'ZigZag':
        return ZigZag(d=d, regularization=reg, seed=seed)
    elif fun_name == 'ZigZagSmooth':
        return ZigZagSmooth(d=d, regularization=reg, seed=seed)
    elif fun_name == 'SumExp':
        return SumExpTarget(d=d, regularization=reg, seed=seed)
    raise ValueError(f"Function {fun_name} not recognized")