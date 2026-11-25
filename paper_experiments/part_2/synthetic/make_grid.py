import argparse as ap
import itertools
import os

import numpy as np

L_FRACS = (0.1, 0.3, 0.5, 0.7, 1.0)


MIN_PARTICLES = 10

GAMMAS = np.logspace(-5, 1, 20)
HS = [1e-5]
BETAS = [0.001, 0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0] #np.logspace(-3, 3, 20)


def counts(d, fracs, minimum=1):
    return sorted({1} | {max(minimum, int(round(f * d))) for f in fracs})


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]

def langevin_grid(d):
    return grid(gamma=GAMMAS, beta=BETAS, h=HS, l=counts(d, L_FRACS), s=counts(d, L_FRACS))


def to_flags(cfg):
    return ' '.join(f"--{k.replace('_', '-')} {v}" for k, v in cfg.items())


def main():
    p = ap.ArgumentParser(description="Generate the grid-search job list")
    p.add_argument("target")
    p.add_argument("d", type=int)
    p.add_argument("--reps", default=10, type=int)
    p.add_argument("--budget", default=100000, type=int)
    p.add_argument("--base-seed", default=1234, type=int)
    p.add_argument("--dtype", default='float32', choices=['float32', 'float64'])
    p.add_argument("--device", default='cpu', choices=['cpu', 'cuda'])
    p.add_argument("--out-root", default="./results/grid", help="Root of the per-job output directories")
    p.add_argument("--out", default=None, help="Job list file (default: grid_<algo>_<target>_d<d>.csv)")
    args = p.parse_args()

    out = args.out or f"grid_szlan_{args.target}_d{args.d}.csv"
    common = (f"--reps {args.reps} --budget {args.budget} --base-seed {args.base_seed} "
              f"--dtype {args.dtype} --device {args.device}")

    cfgs = langevin_grid(args.d)
    with open(out, 'w') as f:
        for i, cfg in enumerate(cfgs):
            out_dir = os.path.join(args.out_root, f"szlan_{args.target}_d{args.d}")
            flags = f"{common} --out-dir {out_dir} {to_flags(cfg)}"
            f.write(f"szlan;{args.target};{args.d};{flags}\n")
    print(f"{len(cfgs)} configurations written to {out}")


if __name__ == '__main__':
    main()