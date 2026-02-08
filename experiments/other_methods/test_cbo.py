import torch 


from cbo import CBO
from pso import PSO
from random_search import RS
from emna import EMNA
from de import DifferentialEvolution
import sys 

import tqdm

sys.path.append("../")
from synthetic_functions_torch import ZigZag

import matplotlib.pyplot as plt

d = 50
n = 10
dtype = torch.float64
device = "cpu"

target = ZigZag(d = d, dtype=dtype, device=device)

seed = 12314
generator = torch.Generator(device=device).manual_seed(seed)

population = (target.bounds[:, 1] - target.bounds[:, 0]) * torch.rand((n, d), generator=generator, device=device, dtype=dtype) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)

#opt = CBO(population=population, dt=0.01, lam=0.001, alpha=10.0, sigma=10.0,  seed=seed, dtype=dtype, device=device)

#opt = PSO(population=population, inertia=0.80, c1=1.45, c2=1.45, seed=seed, dtype=dtype, device=device)

#opt = RS(population=population, sigma=0.1, seed=seed, dtype=dtype, device=device)

#opt = DifferentialEvolution(population=population, F=0.8, CR=0.7, seed=seed, dtype=dtype, device=device)

opt = EMNA(population=population, mu=0.5, isotropic=True, seed=seed, dtype=dtype, device=device)

print(target(target.x0))

iterator = tqdm.tqdm(range(50000))

values = []

for k in iterator:
    X = opt.ask()
    Y = target(X)

    iterator.set_postfix({"values": torch.min(Y).item(), "min f": target.min_f})
    opt.tell(X, Y)
    values.append(torch.min(Y).item())
#    print(opt.current_cons_point.flatten())


fig, ax = plt.subplots()
ax.plot(values)
fig.savefig("pso_test.png")