import argparse as ap
import os
import sys

import numpy as np
import torch

sys.path.append("../../")
from other_methods.cbo import CBO
from other_methods.cmaes import CMAES
from other_methods.de import DifferentialEvolution as DE
from other_methods.emna import EMNA
from other_methods.pso import PSO
from other_methods.random_search import RS
from other_methods.xnes import XNES
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
    parser = ap.ArgumentParser(description='Comparison between Zeroth-order methods for non-convex optimization')
    
    parser.add_argument("algorithm", default='zlan', choices=['zlan', 'cbo', 'pso', 'de', 'cmaes', 'xnes', 'emna', 'fd', 'rs'])
    parser.add_argument("fun_name", default='least_squares', choices=['least_squares', 'rastrigin', 'rosenbrock', 'griewank', 'ackley', 'levy'], help='Target function')
    parser.add_argument("--d", default=10, type=int, help='Dimension')

    # Zeroth-order Langevin parameters
    parser.add_argument("--gamma", default=0.01, type=float, help='Stepsize')
    parser.add_argument("--beta", default=1.0, type=float, help='Exploration parameter')
    parser.add_argument("--s", default=1, type=int, help='Number of directions')
    parser.add_argument("--dir-type", default='spherical', choices=['spherical', 'gaussian'], help='Type of directions')
    parser.add_argument("--h", default=1e-5, type=float, help='smoothing parameter')
    
    # CBO/RS parameters
    parser.add_argument("--popsize", default=10, type=int, help='Population size (for CBO, PSO, RS)')
    parser.add_argument("--lam", default=1.0, type=float, help='Population size (for CBO, PSO)')
    parser.add_argument("--alpha", default=10.0, type=float, help='Population size (for CBO, PSO)')
    parser.add_argument("--sigma", default=1.0, type=float, help='Sigma parameter (i.e. CBO: weight of Brownian motion; RS: standard deviation of Gaussian distribution)')
    parser.add_argument("--dt", default=0.01, type=float, help='Time discretization (for CBO)')

    # PSO parameters
    parser.add_argument("--inertia", default=0.729, type=float, help='Inertia parameter (PSO)')
    parser.add_argument("--c1", default=1.49445, type=float, help='Inertia parameter (PSO)')
    parser.add_argument("--c2", default=1.49445, type=float, help='Inertia parameter (PSO)')
    
    # DE parameters
    parser.add_argument("--F", default=0.8, type=float, help='Differential weight (DE)')
    parser.add_argument("--CR", default=0.9, type=float, help='Cross-over parameter (DE)')
    
    # EMNA parameters
    parser.add_argument("--mu", default=0.5, type=float, help='Fraction of population used to compute updates (EMNA)')
    parser.add_argument("--min-sigma", default=1e-5, type=float, help='Minimum of standard deviation (EMNA)')

    # XNES parameters
    parser.add_argument("--eta_mu", default=1.0, type=float, help='xNES initial stepsize')
    
    
    # Experiments parameters
    parser.add_argument("--budget", default=100000, type=int, help='Number of function evaluations')
    parser.add_argument("--reps", default=5, type=int, help='Repetitions')
    parser.add_argument('--seed', default=5414, type=int, help='Random seed')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'], help='Datatype')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--out-dir', default="./", type=str, help="Directory where results will be stored")
    
    
    return parser.parse_args()


def get_direction_generator(d, l, s, dir_type, dir_seed, dtype, device):
    if dir_type == 'gaussian':
        return GaussianDirectionGenerator(d=d, l=l, s=s, seed=dir_seed, dtype=dtype, device=device)
    elif dir_type == 'spherical':
        return SphericalDirectionGenerator(d = d, l=l, s=s, seed=dir_seed, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown direction type '{args.dir_type}'!")


def build_population(x0, popsize, seed, dtype, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    if popsize > 0:
        population = 5.0 * torch.randn((popsize, x0.shape[0]), dtype=dtype, device=device, generator=generator)
        population = torch.vstack((x0.view(1, -1), population))
    else:
        population = x0.view(1, -1)
    return population

def get_algorithm(args, x0, dtype, device, seed, rep):
    d = args.d
    opt_seed  = 835 * seed + 333
    dir_seed = 127 * seed + 546
    pop_seed = 333 * seed + 321
    
    if args.algorithm in ['zlan' ,'fd']:
        gamma, beta, s, h = args.gamma, args.beta, args.s, args.h  
        
        beta_t = lambda t : beta # * (t + 2) / log(t + 2)
               
        dir_generator = get_direction_generator(d = d, l = 1, s = s, dir_type = args.dir_type, dir_seed = dir_seed, dtype = dtype, device = device)
        
        return SZLan(x0=x0, direction_generator=dir_generator, h = h, gamma=gamma, only_fd=(args.algorithm == 'fd'), beta=beta_t, seed=opt_seed + 12 * rep, device=device, dtype=dtype)

    elif args.algorithm in ['cbo', 'rs', 'pso', 'de', 'cmaes', 'xnes', 'emna']:
        popsize = args.popsize - 1
        sigma = args.sigma
        population = build_population(x0, popsize, pop_seed + rep * 9, dtype=dtype, device=device)
        
        if args.algorithm == 'cbo':                    
            dt, lam, alpha = args.dt, args.lam, args.alpha
            return CBO(population=population, dt = dt, lam = lam, alpha = alpha, sigma = sigma, seed=opt_seed, dtype=dtype, device=device)
        elif args.algorithm == 'pso':
            inertia, c1, c2 = args.inertia, args.c1, args.c2
            return PSO(population=population, inertia=inertia, c1=c1, c2=c2, seed=opt_seed, dtype=dtype, device=device)
        elif args.algorithm == 'de':
            F, CR = args.F, args.CR
            return DE(population=population, F = F, CR=CR, seed=opt_seed, dtype=dtype, device=device)
        elif args.algorithm == 'cmaes':
            return CMAES(population, sigma=sigma, seed=opt_seed, dtype=dtype, device=device)#, mu=x0)
        elif args.algorithm == 'xnes':
            eta_mu = args.eta_mu
            return XNES(population, sigma=sigma, eta_mu=eta_mu, seed=opt_seed, dtype=dtype, device=device)
        elif args.algorithm == 'emna':
            mu, min_sigma = args.mu, args.min_sigma
#    def __init__(self, population, mu = 0.5, isotropic = True, min_sigma=1e-5, seed=123123, dtype=torch.float64, device='cpu'):
            return EMNA(population=population, mu=mu, min_sigma=min_sigma, seed=opt_seed, dtype=dtype, device=device)
        return RS(population = population, sigma=sigma, seed=opt_seed, dtype=dtype, device=device)

    raise ValueError(f"Unrecognised algorithm {args.algorithm}!")

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
        target=Levy(d = d, dtype = dtype, device = device)
        return target
    raise ValueError(f"Unrecognised function name {fun_name}!")


def run_experiment(target, gamma, beta, direction_type, s, h, reps, T, args, seed, dtype, device):
    
    init_seed = 422 * seed + 131
    
    function_values = [[] for _ in range(reps)]
    
    init_generator = torch.Generator(device).manual_seed(init_seed)
    
    for rep in range(reps):
        
        x0 = (target.bounds[1] - target.bounds[0]) * torch.rand((target.d, ), dtype=dtype, device=device, generator=init_generator) + target.bounds[0]
        y0 = target(x0)
        
        opt = get_algorithm(args, x0, dtype, device, seed, rep)

        eval_f = torch.vmap(lambda x: target(x), in_dims=(0,))
        num_evaluations = 0

        while num_evaluations < T:
            x = opt.ask()            
            y = eval_f(x) 
            
            
            
            if torch.isnan(y).any() or torch.isinf(y).any() or torch.any(y /y0  > 1e10):
                return np.array([1.0 for _ in range(reps)]), np.array([0.0 for _ in range(reps)]), True

            opt.tell(x, y)
            
            y_val = y.min()
            
            function_values[rep].append(y_val.item())
            num_evaluations += y.flatten().shape[0]
            print(f"[--] rep = {rep}/{reps}, k = {num_evaluations}/{T}, F_k / F_0= {function_values[rep][-1]/function_values[rep][0]:.5f}, best: {min(function_values[rep])}")
    
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
    out_dir = f"{base_out_dir}/zlan_results/comparison"
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

    mu_values, std_values, _ = run_experiment(target, gamma, beta, direction_type, s, h, reps, T,args,  base_seed, dtype, device)


    with open(f"{out_dir}/traces/{args.algorithm}_{fun_name}_{d}.csv", 'w') as f:
        for i in range(mu_values.shape[0]):
            f.write(f"{mu_values[i]},{std_values[i]}\n")
            f.flush()


if __name__ == '__main__':
    args = get_args()
    main(args)