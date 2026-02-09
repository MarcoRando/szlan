import torch 


from cbo import CBO
from pso import PSO
from random_search import RS
from emna import EMNA
from de import DifferentialEvolution
import sys 

from math import sqrt, log

import tqdm

sys.path.append("../")
from synthetic_functions_torch import ZigZag, AckleyFunction, StyblinksiTangFunction, TridFunction

sys.path.append("../../")
from szlan.optimizer.szlan import SZLan
from szlan.direction_generators.direction_generators import QRDirectionGenerator

import matplotlib.pyplot as plt

d = 500
s = 10
n = 100
dtype = torch.float64
device = "cpu"

target = AckleyFunction(d = d, dtype=dtype, device=device)

seed = 12314
generator = torch.Generator(device=device).manual_seed(seed)

population = (target.bounds[:, 1] - target.bounds[:, 0]) * torch.rand((n, d), generator=generator, device=device, dtype=dtype) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)


max_beta, min_beta = 10.0, 1e-3
betas = (max_beta - min_beta) * torch.rand((n, 1), generator=generator, device=device, dtype=dtype)  + min_beta

gamma = 0.001
beta = lambda k : 10.0 # gamma * (k + 2)#/ log(k + 2) 
h = lambda k : 1e-3# / (k + 1) #100.0 / ((k + 1)**3)


optimizers = {
    # 'cbo' : CBO(population=population.clone(), dt=0.001, lam=0.01, alpha=100.0, sigma=5.0,  seed=seed, dtype=dtype, device=device),
    # 'pso' : PSO(population=population.clone(), inertia=0.80, c1=1.45, c2=1.45, seed=seed, dtype=dtype, device=device),
    # 'rs' :  RS(population=population.clone(), sigma=0.1, seed=seed, dtype=dtype, device=device),
    # 'de' :  DifferentialEvolution(population=population.clone(), F=0.8, CR=0.7, seed=seed, dtype=dtype, device=device),
    # 'emna' : EMNA(population=population.clone(), mu=0.5, isotropic=True, seed=seed, dtype=dtype, device=device),
    # 'ozd' : SZLan(population=population.clone(), direction_generator=QRDirectionGenerator(d=d, l=d, s=n, seed=seed, dtype=dtype, device=device), gamma=0.001, just_fd=True, h=1e-3, seed=seed, dtype=dtype, device=device),
    'szlan' : SZLan(population=population.clone(), direction_generator=QRDirectionGenerator(d=d, l=s, s=s, seed=seed, dtype=dtype, device=device), gamma=gamma, beta=beta, h=h, seed=seed, dtype=dtype, device=device)
}

fig, ax = plt.subplots()

budget = 100000000

for (label, opt) in optimizers.items():

    print(f"[--] Optimizing with {label.upper()}")

    iterator = tqdm.tqdm(range(1000))
    num_evals = 0
    values = []
    best_fx = None
    while num_evals < budget:
        X = opt.ask()
        Y = target(X)
        num_evals += X.shape[0]
        print(f"[{label}]Values: {torch.min(Y).item()}, min f: {target.min_f}, num_evals: {num_evals}/{budget}")

        opt.tell(X, Y)
        if best_fx is None or torch.min(Y).item() < best_fx:
            best_fx = torch.min(Y).item()
        values.append(best_fx)

    #    print(opt.current_cons_point.flatten())


    ax.plot(range(1, len(values) + 1), values, lw=3, label=label)
ax.set_xscale("log")
ax.legend()
fig.savefig("opt_test.png")