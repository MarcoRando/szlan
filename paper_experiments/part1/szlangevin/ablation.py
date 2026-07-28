import argparse as ap
import fcntl
import os
import sys

import numpy as np
import torch
import tqdm

sys.path.append("../../")
from synthetic_functions import (
    Ackley,
    Griewank,
    LeastSquares,
    Levy,
    Rastrigin,
    Rosenbrock,
)

sys.path.append("../../../")
from szlan.direction_generators.direction_generators import (
    GaussianDirectionGenerator,
    SphericalDirectionGenerator,
)
from szlan.optimizer.szlan import SZLan

DTYPES = {
    'float32' : torch.float32,
    'float64' : torch.float64
}


def get_args():
    parser = ap.ArgumentParser(description='Ablation study for Zeroth-order Langevin algorithm')
    parser.add_argument("fun_name", default='least_squares', choices=['least_squares', 'rastrigin', 'rosenbrock', 'griewank', 'ackley', 'levy'], help='Target function')
    parser.add_argument("--d", default=10, type=int, help='Dimension')

    # Zeroth-order Langevin parameters
    parser.add_argument("--gamma", default=0.01, type=float, help='Stepsize')
    parser.add_argument("--beta", default=1.0, type=float, help='Exploration parameter')
    parser.add_argument("--s", default=1, type=int, help='Number of directions')
    parser.add_argument("--dir-type", default='spherical', choices=['spherical', 'gaussian'], help='Type of directions')
    parser.add_argument("--h", default=1e-5, type=float, help='smoothing parameter')
    
    # Experiments parameters
    parser.add_argument("--budget", default=1000000, type=int, help='Number of function evaluations')
    parser.add_argument("--reps", default=5, type=int, help='Repetitions')
    parser.add_argument('--seed', default=5414, type=int, help='Random seed')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'], help='Datatype')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--out-dir', default="./", type=str, help="Directory where results will be stored")
    
    
    return parser.parse_args()


def get_target(fun_name : str, d : int, seed : int, dtype : torch.dtype, device : str):
    if fun_name == 'least_squares':
        return LeastSquares(d = d, L =100.0, mu = 1.0, seed = seed, dtype = dtype, device = device)
    elif fun_name == 'rosenbrock':
        return Rosenbrock(d = d, dtype = dtype, device = device)
    elif fun_name == 'ackley':
        return Ackley(d = d, lam=1e-3, dtype = dtype, device = device)
    elif fun_name == 'rastrigin':
        return Rastrigin(d = d, lam=1e-3, dtype = dtype, device = device)
    elif fun_name == 'griewank':
        return Griewank(d = d, dtype = dtype, device = device)
    elif fun_name == 'levy':
        return Levy(d = d, dtype = dtype, device = device)
        
    raise ValueError(f"Unrecognised function name {fun_name}!")


def run_experiment(target, gamma, beta, direction_type, s, h, reps, T, seed, dtype, device):
    
    init_seed = 422 * seed + 131
    opt_seed  = 835 * seed + 333
    dir_seed = 127 * seed + 546

    function_values = [[] for _ in range(reps)]
    
    init_generator = torch.Generator(device).manual_seed(init_seed)
    
    for rep in range(reps):
        
        x0 = (target.bounds[1] - target.bounds[0]) * torch.rand((target.d, ), dtype=dtype, device=device, generator=init_generator) + target.bounds[0]

        y0 = target(x0)

        if direction_type == 'gaussian':
            dir_generator = GaussianDirectionGenerator(d=target.d, l=1, s=s, seed=dir_seed, dtype=dtype, device=device)
        elif direction_type == 'spherical':
            dir_generator = SphericalDirectionGenerator(d = target.d, l=1, s=s, seed=dir_seed, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown direction type '{direction_type}'!")

        opt = SZLan(x0=x0, direction_generator=dir_generator, h = h, gamma=gamma, beta=beta, seed=opt_seed + 12 * rep, device=device, dtype=dtype)

        iterator = tqdm.tqdm(range(T // (s + 1) ), desc=target.name)
        eval_f = torch.vmap(lambda x: target(x), in_dims=(0,))
        for k in iterator:
            x = opt.ask()            
            y = eval_f(x) 
            
            if torch.isnan(y).any() or torch.isinf(y).any() or torch.any(y /y0  > 1e10):
                return np.array([1.0 for _ in range(reps)]), np.array([0.0 for _ in range(reps)]), True

            opt.tell(x, y)
            function_values[rep].append(y[-1].item())
            iterator.set_postfix({'rep' : f"{rep}/{reps}", 'F_k / F_0' : f"{function_values[rep][-1]/function_values[rep][0]:.5f}"})
    
    function_values = np.array(function_values).reshape(reps, -1) - target.f_star
    function_values /= function_values[:, 0].reshape(-1, 1)
    return np.mean(function_values, axis=0), np.std(function_values, axis=0), False
    
    
    



def main(args):
    fun_name, d = args.fun_name, args.d
    
    direction_type = args.dir_type
    
    gamma, beta, h, s = args.gamma, args.beta, args.h, args.s
    
    T, reps = args.budget, args.reps
    base_seed = args.seed
    dtype, device = DTYPES[args.dtype], args.device
    
    base_out_dir = args.out_dir
    out_dir = f"{base_out_dir}/zlan_results/ablation"
    os.makedirs(out_dir + "/traces", exist_ok=True)

    target_seed = 98 * base_seed + 901
    
    if os.path.exists(f"{out_dir}/{fun_name}_{d}_{direction_type}_tested.log"):
        with open(f"{out_dir}/{fun_name}_{d}_{direction_type}_tested.log", 'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            for line in lines:
                splitted = line.split(',')
                if gamma == float(splitted[0]) and beta == float(splitted[1]) and s==int(splitted[2]):
                    print("[--] This experiment has been already performed!")
                    return

    target = get_target(fun_name, d, target_seed, dtype, device)

    mu_values, std_values, _ = run_experiment(target, gamma, beta, direction_type, s, h, reps, T, base_seed, dtype, device)


    num_iters = T // (s + 1)

    with open(f"{out_dir}/{fun_name}_{d}_{direction_type}_tested.log", 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        idx_best = np.argmin(mu_values + std_values)
        lst_500_mu, lst_500_std = mu_values[-100:], std_values[-100:]
        best_last_500 = np.argmin(lst_500_mu + lst_500_std)

        f.write(f"{gamma},{beta},{s},{mu_values[idx_best]},{std_values[idx_best]},{idx_best},{lst_500_mu[best_last_500]},{lst_500_std[best_last_500]},{num_iters - 100 + best_last_500}\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
        


if __name__ == '__main__':
    args = get_args()
    main(args)