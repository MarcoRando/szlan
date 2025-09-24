import os
import numpy as np 
import argparse as ap 

import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import Optimizer, SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import GaussianDirectionGenerator, SphericalDirectionGenerator, QRDirectionGenerator

from synthetic_functions import *
from utils import get_objective_function

from comparisons_utils import get_optimizer, get_params_grid

import nevergrad as ng

from itertools import product
from functools import partial
import multiprocessing as mp 






def run_optimizer(params, optimizer_name, target, budget, d, h, seed):
#    l, gamma, beta, n, rep = params
    rep = params[-1]
    n = params[-2]
    budget = budget #* n

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep


    rnd_state = np.random.RandomState(seed+ 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)


    np.random.seed(opt_seed)
    opt = get_optimizer(optimizer_name, params, population, d, h, seed)


    fvalues = []
    costs =[]

    if not isinstance(opt, Optimizer):
        opt = opt(parametrization=d, budget=budget )
        for vec in population:
            cand = opt.parametrization.spawn_child(new_value=vec)
            fx = target(vec.reshape(1, -1))
            fvalues.append(fx)
            opt.tell(cand, fx)
            budget-=1
            costs.append(1)

    # direction_generator = QRDirectionGenerator(d=d, l=l, seed=direction_seed)
    # opt = SZLan(population=population, gamma=gamma, beta=beta , h=h, direction_generator=direction_generator, seed=opt_seed)

#    print("------------")
#    print(f"Running {optimizer_name} with parameters: {params}....")
#    print("------------")

    num_evals = 0

    best_on_iterate = None
   # fvalues = []
#    print("BEFORE LOOP -> ", budget)
    while num_evals < budget:
        X = opt.ask()
#        Y = target(X)
        Y = target(X.value.reshape(-1, d)) if not isinstance(X, np.ndarray) else target(X.reshape(-1, d))

        if np.any(np.isnan(Y)):
            print("NAN")
            return np.nan, np.nan, fvalues + [np.nan], costs + [np.nan], params
            #            f_found, f_found_on_iterate, fvalues, costs, params = result


#        if not isinstance(opt, Optimizer):

#        if  opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE:
#        print(Y, np.min(Y))
        fvalues.append(np.min(Y))
#        if opt.phase == SZLanPhase.ITERATE and (best_on_iterate is None or np.min(Y) < best_on_iterate):
        best_on_iterate = np.min(Y)      
        opt.tell(X, Y)

        cost = 1 if not isinstance(opt, Optimizer) else  X.shape[0]
        num_evals += cost
        costs.append(cost)

    f_best = target(opt.recommend().value.reshape(1, -1))[0] if not isinstance(opt, Optimizer) else target(opt.recommend()[0].reshape(1, -1))[0]
#    print("ALL VALUES")
#    print(fvalues)
    return f_best, best_on_iterate, fvalues, costs, params




def run_experiment(args):
    optimizer_name = args.optimizer
    target_name = args.fun_name
    d = args.d
    h = args.h
    reg = args.reg
    budget = args.budget #* (d + 1)
    seed = args.seed
    num_workers = args.num_workers
    reps = args.reps
    output_directory = f"{args.out_dir}/szlan_results/comparison/{target_name}_{reg}/{optimizer_name}"
    os.makedirs(output_directory + "/traces", exist_ok=True)
    np.random.seed(seed)

    param_grid = get_params_grid(optimizer_name, d, reps)

    filtered_param_grid = []
    for param in param_grid:
        print(param)
        param_string = "_".join([str(x) for x in param])
        if not os.path.exists(f"{output_directory}/traces/{optimizer_name}_{target_name}_{d}_{param_string}_trace.txt"):
            filtered_param_grid.append(param)
        else:
            print(f"Skipping configuration {param}")
    param_grid = filtered_param_grid

    target = get_objective_function(target_name, d=d, reg=reg, seed=seed)
    target.regularization = 0.0
    min_f = target(target.x_star)[0]
    target.regularization = reg
    min_f_reg = target(target.x_star)[0]
    print(f"Function minimum: {min_f}")

    run_opt = partial(run_optimizer, optimizer_name=optimizer_name, target=target, budget=budget, d=d, h=h, seed=seed)
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( run_opt, param_grid):
            f_found, f_found_on_iterate, fvalues, costs, params = result
            rep = params[-1]
            out_file_sumup = f"{optimizer_name}_{target_name}_{d}_" + "_".join([str(x) for x in params[:-1]]) + ".txt"
            out_file_trace = f"{optimizer_name}_{target_name}_{d}_" + "_".join([str(x) for x in params]) + "_trace.txt"
            print(f"[{optimizer_name}] {params} f found: {f_found}\trep: {rep}\tmin f: {min_f}")
            with open(f"{output_directory}/{out_file_sumup}", "a") as f:
                f.write(f"{rep},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f_reg},{min_f}\n")
                f.flush()
            with open(f"{output_directory}/traces/{out_file_trace}", "a") as f:
                for i in range(len(fvalues)):
                    f.write(f"{fvalues[i]},{fvalues[0]},{costs[i]},{min_f_reg},{min_f}\n")
                    f.flush()

    




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    
    # Experiment Parameters
    parser.add_argument("optimizer", type=str, default="szlan", choices=["szlan", 'fd_gaus', 'fd_sph', 'fd_orth', "cmaes", "de_2p", "pso", 'cbo'], help="Optimizer to use")
    parser.add_argument("fun_name", type=str, default="Rosenbrock", choices=["Rosenbrock", "Griewank", "Ackley", "Levy", "StyblinkskiTang", "Rastrigin", "Quing"], help="Objective function to optimize")
    parser.add_argument("--d", type=int, default=10, help="Dimension of the problem")
    parser.add_argument('--budget', type=int, default=1000, help='Function evaluation budget')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions for the experiment')
    parser.add_argument('--reg', type=float, default=0.0, help='Regularization parameter')

    # Optimizer Parameters
    parser.add_argument('--h', type=float, default=1e-7, help='Finite difference step size')

    # General Parameters
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    args = parser.parse_args()
    run_experiment(args)
    
    