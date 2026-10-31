import argparse as ap
import itertools
import os

ALGOS = ['szd', 'zd', 'szlan', 'zlan', 'rs', 'cbo', 'pso']

# Fractions of the dimension d
L_FRACS = (0.1, 0.5, 1.0)       # number of directions
B_FRACS = (0.1, 0.5, 1.0)       # random-search batch size
P_FRACS = (0.25, 0.5, 1.0)      # swarm size (CBO / PSO)
MIN_PARTICLES = 10

GAMMAS = (1e-3, 1e-2, 1e-1)
HS = (1e-3, 1e-5)
BETAS = (10.0, 100.0)


def counts(d, fracs, minimum=1):
    """Distinct integer counts max(minimum, round(frac * d))."""
    return sorted({max(minimum, int(round(f * d))) for f in fracs})


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


def fd_grid(d):
    return grid(gamma=GAMMAS, h=HS, l=counts(d, L_FRACS))


def langevin_grid(d):
    return grid(gamma=GAMMAS, beta=BETAS, h=HS, l=counts(d, L_FRACS))


def rs_grid(d):
    return grid(sigma=(0.01, 0.1, 1.0), batch_size=counts(d, B_FRACS))


def cbo_grid(d):
    return grid(n_particles=counts(d, P_FRACS, MIN_PARTICLES), sigma=(0.1, 1.0),
                alpha=(10.0, 100.0), init_std=(1.0, 3.0))


def pso_grid(d):
    cfgs = grid(n_particles=counts(d, P_FRACS, MIN_PARTICLES), init_std=(1.0, 3.0),
                w=(0.5, 0.7298), c=(1.0, 1.49618))
    for cfg in cfgs:                      # c1 = c2 = c
        cfg['c1'] = cfg['c2'] = cfg.pop('c')
    return cfgs


GRIDS = {'szd': fd_grid, 'zd': fd_grid, 'szlan': langevin_grid, 'zlan': langevin_grid,
         'rs': rs_grid, 'cbo': cbo_grid, 'pso': pso_grid}


def to_flags(cfg):
    return ' '.join(f"--{k.replace('_', '-')} {v}" for k, v in cfg.items())


def main():
    p = ap.ArgumentParser(description="Generate the grid-search job list")
    p.add_argument("algo", choices=ALGOS)
    p.add_argument("target")
    p.add_argument("d", type=int)
    p.add_argument("--reps", default=10, type=int)
    p.add_argument("--budget", default=10000, type=int)
    p.add_argument("--base-seed", default=1234, type=int)
    p.add_argument("--dtype", default='float32', choices=['float32', 'float64'])
    p.add_argument("--device", default='cuda', choices=['cpu', 'cuda'])
    p.add_argument("--out-root", default="./results/grid", help="Root of the per-job output directories")
    p.add_argument("--out", default=None, help="Job list file (default: grid_<algo>_<target>_d<d>.csv)")
    args = p.parse_args()

    out = args.out or f"grid_{args.algo}_{args.target}_d{args.d}.csv"
    common = (f"--reps {args.reps} --budget {args.budget} --base-seed {args.base_seed} "
              f"--dtype {args.dtype} --device {args.device}")

    cfgs = GRIDS[args.algo](args.d)
    with open(out, 'w') as f:
        for i, cfg in enumerate(cfgs):
            out_dir = os.path.join(args.out_root, f"{args.algo}_{args.target}_d{args.d}", f"cfg{i:03d}")
            flags = f"{common} --out-dir {out_dir} {to_flags(cfg)}"
            f.write(f"{args.algo};{args.target};{args.d};{flags}\n")
    print(f"{len(cfgs)} configurations written to {out}")


if __name__ == '__main__':
    main()