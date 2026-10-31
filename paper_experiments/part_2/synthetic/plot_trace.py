import argparse as ap
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COMMON_PARAMS = {'algorithm', 'target', 'd', 'budget', 'reps', 'base_seed', 'dtype'}


def get_arguments():
    parser = ap.ArgumentParser(description="Plot the best traces of a function for a given dimension")
    parser.add_argument("target", type=str, help="Target function name")
    parser.add_argument("d", type=int, help="Dimension")
    parser.add_argument("--out-dir", default="./results", type=str, help="Results directory used by the experiments (contains synthetic/<target>/)")
    parser.add_argument("--budget", default=None, type=int, help="Only plot traces with this budget")
    parser.add_argument("--stat", default="median", choices=["median", "mean"], help="Central line")
    parser.add_argument("--band", default="percentile", choices=["percentile", "ci", "std", "none"],
                        help="Shaded region: p05-p95, mean +- 95%% CI, mean +- std, or none")
    parser.add_argument("--linear-y", action="store_true", help="Linear y axis (default: log)")
    parser.add_argument("--log-x", action="store_true", help="Log x axis")

    return parser.parse_args()


def read_trace(path):
    with open(path) as fin:
        first = fin.readline()
    has_params = first.startswith("# params:")
    params = json.loads(first[len("# params:"):]) if has_params else {}
    data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1 if has_params else 0)
    return params, data


def make_label(params, path):
    if not params:
        return os.path.basename(path)
    extra = ", ".join(f"{k}={v}" for k, v in params.items() if k not in COMMON_PARAMS)
    return f"{params.get('algorithm', '?')} ({extra})" if extra else params.get('algorithm', '?')


def main(args):
    folder = os.path.join(args.out_dir, "synthetic", args.target)
    pattern = os.path.join(folder, f"*_d{args.d}_b*_best_trace.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No trace files found matching {pattern}")

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for path in files:
        params, data = read_trace(path)
        if args.budget is not None and params.get('budget') != args.budget:
            continue
        x = data['evaluations']
        center = data[args.stat]
        label = make_label(params, path)
        if params.get('budget') is not None and args.budget is None:
            label += f" [budget={params['budget']}]"

        (line,) = ax.plot(x, center, label=label, rasterized=True)
        if args.band == "percentile":
            lo, hi = data['p05'], data['p95']
        elif args.band == "ci":
            lo, hi = data['mean'] - data['ci95_halfwidth'], data['mean'] + data['ci95_halfwidth']
        elif args.band == "std":
            lo, hi = data['mean'] - data['std'], data['mean'] + data['std']
        else:
            lo = hi = None
        if lo is not None:
            if not args.linear_y:
                lo = np.maximum(lo, np.finfo(float).tiny)  # keep the band valid on a log axis
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, rasterized=True)
        plotted += 1

    if plotted == 0:
        raise SystemExit("No trace matches the requested budget")

    ax.set_xlabel("function evaluations")
    ax.set_ylabel("normalized best value")
    ax.set_title(f"{args.target}, d={args.d}")
    if not args.linear_y:
        ax.set_yscale("log")
    if args.log_x:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(f"./trace_{args.target}_{args.d}.pdf", dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main(get_arguments())