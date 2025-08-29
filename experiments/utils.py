import numpy as np 

from synthetic_functions import RosenbrockFunction, LeastSquares, QuingFunction, GriewankFunction


def get_objective_function(fun_name, d, seed):
    if fun_name == "Rosenbrock":
        return RosenbrockFunction(d=d, seed=seed)
    raise ValueError(f"Function {fun_name} not recognized")