import numpy as np 
import nevergrad as ng

from itertools import product
import sys 

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../other_methods")
from szlan.optimizer.szlan import Optimizer, SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import GaussianDirectionGenerator, SphericalDirectionGenerator, QRDirectionGenerator


from other_methods.cbo import CBO
from other_methods.lan import Langevin




def get_optimizer(optimizer_name, params, population, d, h, seed):

    if optimizer_name in ["szlan", "fd_gaus", "fd_sph", "fd_orth"]:
        l, gamma, beta, dir_type, rep = params
        direction_seed = seed + 134 * rep
        opt_seed = seed + 473 * rep
        if dir_type == 'gaussian':
            direction_generator = GaussianDirectionGenerator(d=d, l=l, seed=direction_seed)
        elif dir_type == 'spherical':
            direction_generator = SphericalDirectionGenerator(d=d, l=l, seed=direction_seed)
        else:
            direction_generator = QRDirectionGenerator(d=d, l=l, seed=direction_seed)
        return SZLan(population=population, gamma=gamma, beta= beta , h=h, direction_generator=direction_generator, seed=opt_seed)
    elif optimizer_name == "lan":
        gamma, beta, grad_fun, rep = params
        return Langevin(population=population, gamma=gamma, beta= beta , grad_fun=grad_fun, seed=opt_seed)
    elif optimizer_name == "cmaes":
        return ng.optimizers.ParametrizedCMA(popsize=population.shape[0], elitist=True, fcmaes=True)#, scale=params[0])
    elif optimizer_name == 'cbo':
        dt, lam, alpha, sigma,use_cons, rep = params
        opt_seed = seed + 473 * rep
        return CBO(population=population, dt=dt, lam=lam, alpha=alpha, sigma=sigma, use_cons=use_cons==1, seed=opt_seed)
    elif optimizer_name == "de_2p":
        return ng.optimizers.DifferentialEvolution(popsize=population.shape[0], crossover='twopoints') #(parametrization=d, budget=budget)
    elif optimizer_name == "pso":
        return ng.optimizers.ConfPSO(popsize=population.shape[0],  phip=params[0], phig=params[1])#, omega=params[0], phip=params[1], phig=params[2])
    raise Exception(f"Unknown optimizer {optimizer_name}")




def get_params_grid(optimizer_name, d, reps = 10):
  #  num_particles = [2, 5, 10, 100] #, 10]#, 100, 500] #[2, 5, 10, 100]#, 1000]
    gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0] #np.logspace(-4, 0, 20) #np.logspace(-3, 0, 10) #[0.001, 0.01, 0.05, 0.1, 1.0]#, 1.0]
    num_directions = [1, 2, d//2, d]
    print(f"[--] Getting param grid for {optimizer_name}")
    if optimizer_name == "szlan":


        betas = [0.1, 1.0, 10.0, 50.0, 100.0] #np.logspace(-2, 3, 10)#[0.01, 0.1, 1.0, 5.0, 10.0, 100.0, 1000.0, 5000.0] 
        dir_type = ['qr']
        params = [num_directions, gammas, betas, dir_type]
 #       num_particles.append(1)
    elif optimizer_name in ["fd_gaus", "fd_sph", "fd_orth"]:

        betas = [0.0] 
        if optimizer_name == "fd_gaus":
            dir_type = ['gaussian']
        elif optimizer_name == "fd_sph":
            dir_type = ['spherical']
        else:
            dir_type = ['qr']
        params = [num_directions, gammas, betas, dir_type]
    elif optimizer_name == "cmaes":
        scales = np.logspace(-2, 1, 5) #np.linspace(0.1, 1.0, 5) #[1.0, 0.01, 0.1] #np.linspace(0.1, 1.0, 5)
        params = [scales]
    elif optimizer_name == 'cbo':
        #    def __init__(self, population, dt, lam, alpha, sigma, seed):
        dt = [0.0001, 0.001, 0.01, 0.1, 1.0] #np.linspace(0.01, 1.0, 5) #np.linspace(0.001, 1.0, 20) #np.linspace(0.01, 1.0, 5)
        lam = [0.01, 0.1, 1.0, 10.0, 100.0] #np.linspace(0.1, 1.0, 5) #np.logspace(-2, 1, 20) #np.linspace(0.1, 1.0, 5)
        alpha = [0.001, 0.01, 0.1, 1.0] # np.linspace(0.1, 1.0, 5) #np.logspace(-2, 1, 20) #np.linspace(0.1, 1.0, 5)
        sigma = [0.01, 0.1, 1.0, 10.0, 100.0]# np.linspace(0.1, 10.0, 10) #np.linspace(0.1, 50.0, 20) #np.linspace(0.1, 10.0, 10)
        use_cons = [0] #[0, 1]
        params = [dt, lam, alpha, sigma, use_cons]
    elif optimizer_name == "de_2p":
        scales = np.linspace(0.1, 1.0, 5)  #[0.01, 0.1]# np.linspace(0.1, 1.0, 5)
        diff_weights = np.linspace(0.1, 1.0, 5) #[0.01, 0.1, 1.0] #np.linspace(0.1, 1.0, 5)
        params = [scales, diff_weights, diff_weights]
    elif optimizer_name == "pso":
        omegas = np.linspace(0.01, 1.0, 5) #np.linspace(0.1, 1.0, 5)#[0.001, 0.01, 0.1]#= np.linspace(0.1, 1.0, 5)
        phis = np.linspace(0.1, 2.0, 5) #np.linspace(0.1, 2.0, 5) #[0.01, 0.1, 1.0] #np.linspace(0.1, 2.0, 5)
        params = [omegas, phis, phis]

    params += [range(reps)] #[num_particles, range(reps)]

    return list(product(*params))





def run_optimizer(params, optimizer_name, target, budget, d, h, n, seed):
#    l, gamma, beta, n, rep = params

    rep = params[-1]

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep
    min_f = target(target.x_star)[0]

    rnd_state = np.random.RandomState(seed+ 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)


    np.random.seed(opt_seed)
    opt = get_optimizer(optimizer_name, params, population, d, h, seed)


    fvalues = []
    normalized_fvalues = []
    costs =[]
    num_evals = 0

    if not isinstance(opt, Optimizer):
        opt = opt(parametrization=d, budget=budget )
        for vec in population:
            cand = opt.parametrization.spawn_child(new_value=vec)
            fx = target(vec.reshape(1, -1))
            fvalues.append(fx[0])
            opt.tell(cand, fx)
            num_evals += 1

        fvalues = [np.min(fvalues)]
        costs.append(num_evals)
    if len(fvalues) > 0:
        f_0 = fvalues[0]
        normalized_fvalues.append(1.0)

    while num_evals < budget:
        X = opt.ask()

        Y = target(X.value.reshape(-1, d)) if not isinstance(X, np.ndarray) else target(X.reshape(-1, d))

        if np.any(np.isnan(Y)):
            print("NAN")
            return np.nan, fvalues + [np.nan], costs + [np.nan], params

        fvalues.append(np.min(Y))
        if len(fvalues) ==1:
            f_0 = fvalues[0]
            normalized_fvalues.append(1.0)
        else:
            opt_gap = (fvalues[-1] - min_f) / (f_0 - min_f)
            if normalized_fvalues[-1] > opt_gap:
                normalized_fvalues.append(opt_gap)
            else:
                normalized_fvalues.append(normalized_fvalues[-1])

        opt.tell(X, Y)

        cost = 1 if not isinstance(opt, Optimizer) else  X.shape[0]
        num_evals += cost
        costs.append(cost)

    f_best = target(opt.recommend().value.reshape(1, -1))[0] if not isinstance(opt, Optimizer) else target(opt.recommend()[0].reshape(1, -1))[0]

    return f_best, fvalues, normalized_fvalues, costs, params
