import argparse as ap
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.append("../../../")

from szlan.directions.directions import GENERATORS
from szlan.optimizer.cbo import CBO
from szlan.optimizer.pso import PSO
from szlan.optimizer.rs import RandomSearch
from szlan.optimizer.szlan import SZLan

STRUCTURED = 'structured'
SPHERICAL = 'spherical'


@dataclass
class AlgoSpec:
    add_args: Callable[[ap.ArgumentParser], None]
    params: list
    build: Callable


def _add_szlan_args(parser, with_beta):
    parser.add_argument("--gamma", default=0.1, type=float, help='stepsize')
    if with_beta:
        parser.add_argument("--beta", default=10.0, type=float, help='exploration parameter for langevin')
    parser.add_argument("--l", default=1, type=int, help='Number of directions')
    parser.add_argument("--s", default=1, type=int, help='Number of direction matrices')
    parser.add_argument("--h", default=1e-5, type=float, help='Discretization parameter')


def _build_szlan(args, ctx, only_fd, direction_type):
    base_seed, rep = ctx['base_seed'], ctx['rep']
    direction_seed = 132 * base_seed + 981 + rep
    optimizer_seed = 873 * base_seed + 564 + rep
    direction_generator = GENERATORS[direction_type](
        d=ctx['d'], l=args.l, s=args.s, seed=direction_seed, device=ctx['device'], dtype=ctx['dtype'])
    params = {
        'x0': ctx['x0'], 'gamma': args.gamma, 'h': args.h, 'only_fd': only_fd,
        'direction_generator': direction_generator, 'seed': optimizer_seed,
        'device': ctx['device'], 'dtype': ctx['dtype'],
    }
    if not only_fd:
        params['beta'] = args.beta
    return SZLan(**params)


def _add_rs_args(parser):
    parser.add_argument("--sigma", default=0.1, type=float, help='initial step size')
    parser.add_argument("--batch-size", default=1, type=int, help='candidates evaluated per iteration')


def _build_rs(args, ctx):
    optimizer_seed = 873 * ctx['base_seed'] + 564 + ctx['rep']
    return RandomSearch(x0=ctx['x0'], sigma=args.sigma,
                        batch_size=args.batch_size,
                        seed=optimizer_seed, device=ctx['device'], dtype=ctx['dtype'])


def _add_cbo_args(parser):
    parser.add_argument("--n-particles", default=50, type=int, help='Number of particles in the swarm')
    parser.add_argument("--init-std", default=1.0, type=float, help='Std of the initial swarm around x0')
    parser.add_argument("--lam", default=1.0, type=float, help='Drift strength towards the consensus point')
    parser.add_argument("--sigma", default=1.0, type=float, help='Noise strength')
    parser.add_argument("--alpha", default=10.0, type=float, help='Weight sharpness exp(-alpha f)')
    parser.add_argument("--dt", default=0.1, type=float, help='Euler-Maruyama time step')
    parser.add_argument("--anisotropic", action=ap.BooleanOptionalAction, default=True,
                        help='Component-wise (anisotropic) noise; use --no-anisotropic for isotropic')
    parser.add_argument("--batch-size", default=None, type=int,
                        help='Particles evaluated per ask (default: whole swarm)')


def _build_cbo(args, ctx):
    optimizer_seed = 873 * ctx['base_seed'] + 564 + ctx['rep']
    return CBO(x0=ctx['x0'], n_particles=args.n_particles, init_std=args.init_std,
               lam=args.lam, sigma=args.sigma, alpha=args.alpha, dt=args.dt,
               anisotropic=args.anisotropic, batch_size=args.batch_size,
               seed=optimizer_seed, device=ctx['device'], dtype=ctx['dtype'])


def _add_pso_args(parser):
    parser.add_argument("--n-particles", default=50, type=int, help='Number of particles in the swarm')
    parser.add_argument("--init-std", default=1.0, type=float, help='Std of the initial swarm around x0')
    parser.add_argument("--v-init-std", default=0.0, type=float, help='Std of the initial velocities')
    parser.add_argument("--w", default=0.7298, type=float, help='Inertia weight')
    parser.add_argument("--c1", default=1.49618, type=float, help='Cognitive coefficient')
    parser.add_argument("--c2", default=1.49618, type=float, help='Social coefficient')
    parser.add_argument("--v-max", default=None, type=float, help='Velocity clamp (default: none)')
    parser.add_argument("--batch-size", default=None, type=int,
                        help='Particles evaluated per ask (default: whole swarm)')


def _build_pso(args, ctx):
    optimizer_seed = 873 * ctx['base_seed'] + 564 + ctx['rep']
    return PSO(x0=ctx['x0'], n_particles=args.n_particles, init_std=args.init_std,
               v_init_std=args.v_init_std, w=args.w, c1=args.c1, c2=args.c2,
               v_max=args.v_max, batch_size=args.batch_size,
               seed=optimizer_seed, device=ctx['device'], dtype=ctx['dtype'])


ALGORITHMS = {
    # finite differences only (no Langevin noise)
    'szd': AlgoSpec(lambda p: _add_szlan_args(p, False),
                    ['gamma', 'l', 's', 'h'],
                    lambda a, c: _build_szlan(a, c, only_fd=True, direction_type=STRUCTURED)),
    'zd': AlgoSpec(lambda p: _add_szlan_args(p, False),
                   ['gamma', 'l', 's', 'h'],
                   lambda a, c: _build_szlan(a, c, only_fd=True, direction_type=SPHERICAL)),
    # finite differences + Langevin noise
    'szlan': AlgoSpec(lambda p: _add_szlan_args(p, True),
                      ['gamma', 'beta', 'l', 's', 'h'],
                      lambda a, c: _build_szlan(a, c, only_fd=False, direction_type=STRUCTURED)),
    'zlan': AlgoSpec(lambda p: _add_szlan_args(p, True),
                     ['gamma', 'beta', 'l', 's', 'h'],
                     lambda a, c: _build_szlan(a, c, only_fd=False, direction_type=SPHERICAL)),
    'rs': AlgoSpec(_add_rs_args,
                   ['sigma', 'batch_size'],
                   _build_rs),
    'cbo': AlgoSpec(_add_cbo_args,
                    ['n_particles', 'init_std', 'lam', 'sigma', 'alpha', 'dt', 'anisotropic', 'batch_size'],
                    _build_cbo),
    'pso': AlgoSpec(_add_pso_args,
                    ['n_particles', 'init_std', 'v_init_std', 'w', 'c1', 'c2', 'v_max', 'batch_size'],
                    _build_pso),
}


def add_algorithm_subparsers(parser, parents=()):
    sub = parser.add_subparsers(dest='algorithm', required=True, help='Algorithm to use')
    for name, spec in ALGORITHMS.items():
        p = sub.add_parser(name, parents=list(parents), help=f'{name} algorithm')
        spec.add_args(p)
    return sub


def get_algorithm_params(args):
    return {name: getattr(args, name) for name in ALGORITHMS[args.algorithm].params}


def build_algorithm(args, ctx):
    return ALGORITHMS[args.algorithm].build(args, ctx)