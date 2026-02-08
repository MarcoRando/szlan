import os
import numpy as np 
import argparse as ap 
import torch
import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import  QRDirectionGenerator

from synthetic_functions_torch import *
from utils import get_objective_function

from itertools import product
from functools import partial

import torch.multiprocessing as mp
#import multiprocessing as mp


def run_optimizer(params, target, budget, d, h, seed, dtype, device):
    l, b, gamma, beta, n, rep = params

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep

    rnd_state = torch.Generator(device=device).manual_seed(seed + 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * torch.rand((n, d), generator=rnd_state, device=device, dtype=dtype) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)
    direction_generator = QRDirectionGenerator(d=d, l=l, s=b, device=device, dtype=dtype, seed=direction_seed)
    opt = SZLan(population=population, gamma=gamma, beta=beta , h=h, direction_generator=direction_generator, device=device, dtype=dtype, seed=opt_seed)

    print("------------")
    print(f"Running SZLan with parameters: l={l}, s={b}, gamma={gamma}, beta={beta}, h={h}")
    print("------------")

    num_evals = 0
    fvalues = []

    while num_evals < budget:
        X = opt.ask()
        Y = target(X)
        if torch.any(torch.isnan(Y)):
            return np.nan, fvalues + [np.nan], params
#        return np.nan, fvalues + [np.nan], params

        if opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE:
            fvalues.append(torch.min(Y).item())
            print(f"Phase: {opt.phase}, Iteration: {num_evals}, Values: {torch.min(Y).item()}")

        opt.tell(X, Y)


        num_evals += X.shape[0]

    return opt.best[1], fvalues, params




def run_experiment(args):
    mp.set_start_method("spawn", force=True)
    target_name = args.fun_name
    d = args.d
    h = args.h
    budget = args.budget
    seed = args.seed
    num_workers = args.num_workers
    reps = args.reps

    dtype = torch.float64 if args.dtype=='float64' else torch.float32
    device = torch.device(args.device)

    output_directory = f"{args.out_dir}/szlan_results/changing_parameters/{target_name}"
    os.makedirs(output_directory + "/traces", exist_ok=True)

    num_directions = [ d, int((3/4) * d), d//2, d//5, 1]
    gammas =  np.logspace(-5, 0, 20)[::-1]#[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0] #np.logspace(-7, 0, 5)
    betas = np.logspace(-2, 3,20) #1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5] + np.linspace(0.01, 10.0, 10).tolist() # np.logspace(-7, 0, 8) #np.logspace(-5, 0, 5)
    batch_size = num_directions
    num_particles = [1, 10, 100]
    param_grid = list(product(num_directions, batch_size, gammas, betas, num_particles, range(reps)))

    # filtered_param_grid = []
    # for param in param_grid:
    #     if not os.path.exists(f"{output_directory}/traces/{target_name}_{d}_{param[0]}_{param[1]}_{param[2]}_{param[3]}_{param[4]}_{param[5]}_trace.txt"):
    #         filtered_param_grid.append(param)
    #     else:
    #         print(f"Skipping configuration {param}")
    # param_grid = filtered_param_grid

    target = get_objective_function(target_name, d=d, dtype=dtype, device=device)
    min_f = target(target.x_star)[0]
    print(f"Function minimum: {min_f}")
#    world_size = torch.cuda.device_count()
#    print(f"World size: {world_size}")
#    exit()
    run_opt = partial(run_optimizer, target=target, budget=budget, d=d, h=h, seed=seed, dtype=dtype, device=device)
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( run_opt, param_grid):
            f_found, fvalues, params = result

            l, b, gamma, beta, n, rep = params

            with open(f"{output_directory}/{target_name}_{d}.txt", "a") as f:
                f.write(f"{gamma},{beta},{l},{b},{n},{rep},{f_found},{fvalues[-1]},{fvalues[0]},{min_f}\n")
                f.flush()
            # with open(f"{output_directory}/traces/{target_name}_{d}_{l}_{b}_{gamma}_{beta}_{n}_trace.txt", "a") as f:
            #     f.write("------------\n")
            #     for i in range(len(fvalues)):
            #         f.write(f"{fvalues[i]},{fvalues[0]},{min_f}\n")
            #         f.flush()

    




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    
    # Experiment Parameters
    parser.add_argument("fun_name", type=str, default="ZigZag", choices=["ZigZag", "ZigZagSmooth"], help="Objective function to optimize")
    parser.add_argument("--d", type=int, default=10, help="Dimension of the problem")
    parser.add_argument('--budget', type=int, default=1000, help='Function evaluation budget')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions for the experiment')

    # Optimizer Parameters
    parser.add_argument('--h', type=float, default=1e-7, help='Finite difference step size')

    # General Parameters
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'], help='Floating point precision')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    args = parser.parse_args()
    run_experiment(args)
    
    