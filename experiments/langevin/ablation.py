import os
import sys
import torch
import numpy as np 
import tqdm
import argparse as ap 

import fcntl


sys.path.append("../")
from synthetic_functions import LeastSquares, Ackley, Rosenbrock, Rastrigin, Griewank, Levy

sys.path.append("../../")
from szlan.optimizer.lan import Langevin


DTYPES = {
    'float32' : torch.float32,
    'float64' : torch.float64
}


def get_args():
    parser = ap.ArgumentParser(description='Ablation study for Langevin algorithm')
    parser.add_argument("fun_name", default='least_squares', choices=['least_squares', 'rastrigin', 'rosenbrock', 'griewank', 'ackley', 'levy'], help='Target function')
    parser.add_argument("--d", default=10, type=int, help='Dimension')

    # Langevin parameters
    parser.add_argument("--gamma", default=0.01, type=float, help='Stepsize')
    parser.add_argument("--beta", default=1.0, type=float, help='Exploration parameter')
    
    # Experiments parameters
    parser.add_argument("--num-iters", default=100000, type=int, help='Number of iterations')
    parser.add_argument("--reps", default=5, type=int, help='Repetitions')
    parser.add_argument('--seed', default=12314, type=int, help='Random seed')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'], help='Datatype')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--out-dir', default="./", type=str, help="Directory where results will be stored")
    
    
    return parser.parse_args()


def get_target(fun_name : str, d : int, seed : int, dtype : torch.dtype, device : str):
    if fun_name == 'least_squares':
        return LeastSquares(d = d, L =10000.0, mu = 1.0, seed = seed, dtype = dtype, device = device)
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


def run_experiment(target, gamma, beta, reps, T, seed, dtype, device):
    
    init_seed = 422 * seed + 131
    opt_seed  = 835 * seed + 333

    function_values = [[] for _ in range(reps)]
    
    init_generator = torch.Generator(device).manual_seed(init_seed)
    
    for rep in range(reps):
        
        x0 = (target.bounds[1] - target.bounds[0]) * torch.rand((target.d, ), dtype=dtype, device=device, generator=init_generator) + target.bounds[0]
        opt = Langevin(x0, target.grad, gamma, beta, device = device, dtype = dtype, seed=opt_seed + 12 * rep)
        iterator = tqdm.tqdm(range(T), desc=target.name)
        for k in iterator:
            x = opt.ask()
            y = target(x)
            if torch.isnan(y).any() or torch.isinf(y).any():
                return np.array([1.0 for _ in range(reps)]), np.array([0.0 for _ in range(reps)]), True

            opt.tell(x, y)
            function_values[rep].append(y.item())
            iterator.set_postfix({'rep' : f"{rep}/{reps}", 'F_k / F_0' : f"{function_values[rep][-1]/function_values[rep][0]:.5f}"})
    
    function_values = np.array(function_values).reshape(reps, -1) - target.f_star
    function_values /= function_values[:, 0].reshape(-1, 1)
    return np.mean(function_values, axis=0), np.std(function_values, axis=0), False
    
    
    



def main(args):
    fun_name, d = args.fun_name, args.d
    
    gamma, beta = args.gamma, args.beta
    
    T, reps = args.num_iters, args.reps
    base_seed = args.seed
    dtype, device = DTYPES[args.dtype], args.device
    
    base_out_dir = args.out_dir
    out_dir = f"{base_out_dir}/langevin_results/ablation"
    os.makedirs(out_dir + "/traces", exist_ok=True)

    target_seed = 98 * base_seed + 901
    
    if os.path.exists(f"{out_dir}/{fun_name}_{d}_tested.log"):
        with open(f"{out_dir}/{fun_name}_{d}_tested.log", 'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            for line in lines:
                splitted = line.split(',')
                if gamma == float(splitted[0]) and beta == float(splitted[1]):
                    print("[--] This experiment has been already performed!")
                    return

    target = get_target(fun_name, d, target_seed, dtype, device)
    mu_values, std_values, is_nan = run_experiment(target, gamma, beta, reps, T, base_seed, dtype, device)

    with open(f"{out_dir}/{fun_name}_{d}_tested.log", 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        idx_best = np.argmin(mu_values + std_values)
        lst_500_mu, lst_500_std = mu_values[-500:], std_values[-500:]
        best_last_500 = np.argmin(lst_500_mu + lst_500_std)

        f.write(f"{gamma},{beta},{mu_values[idx_best]},{std_values[idx_best]},{idx_best},{lst_500_mu[best_last_500]},{lst_500_std[best_last_500]},{T - 500 + best_last_500}\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
        

    # with open(f"{out_dir}/traces/{fun_name}_{d}_{gamma}_{beta}.csv", 'w') as f:
    #     for i in range(mu_values.shape[0]):
    #         f.write(f"{mu_values[i]},{std_values[i]}\n")
    #         f.flush()


if __name__ == '__main__':
    args = get_args()
    main(args)