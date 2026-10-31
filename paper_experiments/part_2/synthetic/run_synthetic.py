import argparse as ap
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from synthetic_targets import TARGETS

sys.path.append("../../../")

from algo_utils import add_algorithm_subparsers, build_algorithm, get_algorithm_params

DTYPES = {'float32': torch.float32, 'float64': torch.float64}

STAT_FIELDS = ['mean', 'median', 'std', 'sem', 'ci95_halfwidth', 'min', 'max', 'p05', 'p95']
COMMON_PARAMS = ['algorithm', 'target', 'd', 'budget', 'reps', 'base_seed', 'dtype']


def get_arguments():
    common = ap.ArgumentParser(add_help=False)
    common.add_argument("target", default='ackley', choices=list(TARGETS.keys()), help='Target function')
    common.add_argument("d", default=1, type=int, help="Dimension")
    common.add_argument("--reps", default=10, type=int, help='Number of repetitions')
    common.add_argument("--budget", default=10000, type=int, help='Number of function evaluations')
    common.add_argument("--base-seed", default=1234, type=int, help='Base used to compute random seeds')
    common.add_argument("--dtype", default='float32', choices=list(DTYPES.keys()), help='Data type')
    common.add_argument("--device", default='cuda', choices=['cpu', 'cuda'], help='Device')
    common.add_argument("--out-dir", default="./results", type=str, help="Output directory")

    parser = ap.ArgumentParser(description="Synthetic Experiments Runner")
    add_algorithm_subparsers(parser, parents=[common])
    return parser.parse_args()


def compute_stats(x, axis=0):
    n = x.shape[axis]
    std = np.std(x, axis=axis, ddof=1 if n > 1 else 0)
    sem = std / np.sqrt(n)
    return {
        'mean': np.mean(x, axis=axis),
        'median': np.median(x, axis=axis),
        'std': std,
        'sem': sem,
        'ci95_halfwidth': 1.96 * sem,
        'min': np.min(x, axis=axis),
        'max': np.max(x, axis=axis),
        'p05': np.percentile(x, 5, axis=axis),
        'p95': np.percentile(x, 95, axis=axis),
    }


def read_summary(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return [], []
    with open(path, newline='') as fin:
        reader = csv.DictReader(fin)
        return list(reader.fieldnames or []), list(reader)


def read_best_mean(rows, d, budget):
    """Lowest stored average among runs with same dimension and budget."""
    best = float('inf')
    for row in rows:
        try:
            if int(row['d']) != d or int(row['budget']) != budget:
                continue
            best = min(best, float(row['mean']))
        except (KeyError, ValueError, TypeError):
            continue
    return best


def append_summary(path, row, fieldnames_old, rows_old):
    fields = ['timestamp'] + COMMON_PARAMS
    extra = [k for k in row if k not in fields and k not in STAT_FIELDS]
    fields = fields + extra + STAT_FIELDS
    for k in fieldnames_old:
        if k not in fields:
            fields.insert(len(fields) - len(STAT_FIELDS), k)

    if fieldnames_old and fieldnames_old != fields:
        with open(path, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=fields, restval='')
            writer.writeheader()
            writer.writerows(rows_old)
    new_file = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fields, restval='')
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_trace(path, params, evals, histories):
    st = compute_stats(histories, axis=0)
    tmp = path + ".tmp"
    with open(tmp, 'w', newline='') as fout:
        fout.write("# params: " + json.dumps(params) + "\n")
        writer = csv.writer(fout)
        writer.writerow(['iteration', 'evaluations'] + STAT_FIELDS)
        for i in range(histories.shape[1]):
            writer.writerow([i, int(evals[i])] + [repr(float(st[k][i])) for k in STAT_FIELDS])
    os.replace(tmp, path)  # replaces the trace of the previous best configuration


# ---------------------------------------------------------------------------
def main(args):
    algo_name, target_name = args.algorithm, args.target
    d, reps, budget, base_seed = args.d, args.reps, args.budget, args.base_seed
    dtype, device = DTYPES[args.dtype], args.device

    out_dir = os.path.join(args.out_dir, "synthetic", target_name)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{algo_name}_summary.csv")
    trace_path = os.path.join(out_dir, f"{algo_name}_d{d}_b{budget}_best_trace.csv")

    histories, eval_counts = [], []
    for rep in range(reps):
        init_seed = 213 * base_seed + 154 + rep

        target = TARGETS[target_name](d=d, dtype=dtype, device=device, seed=init_seed)
        x0 = target.sample(1)

        ctx = {'x0': x0, 'd': d, 'rep': rep, 'base_seed': base_seed, 'device': device, 'dtype': dtype}
        opt = build_algorithm(args, ctx)

        eval_performed = 0
        f_0 = target(x0).item()
        min_f = target(target.x_star).item()

        best_f = f_0
        hist, evals = [1.0], [0]
        while eval_performed <= budget:
            points = opt.ask()
            values = target(points)
            opt.tell(points, values)
            eval_performed += points.shape[0]

            best_f = min(best_f, values.min().item())
            hist.append((best_f - min_f) / (f_0 - min_f))
            evals.append(eval_performed)
            print(f"[rep {rep + 1}/{reps}] [{eval_performed}/{budget}] best normalized value: {hist[-1]:.6e} local best: {values.min().item()}")

        histories.append(hist)
        eval_counts.append(evals)

    n_it = min(len(h) for h in histories)
    histories = np.array([h[:n_it] for h in histories], dtype=np.float64)
    evals = np.array(eval_counts[0][:n_it])

    st = compute_stats(histories[:, -1], axis=0)

    params = {'algorithm': algo_name, 'target': target_name, 'd': d, 'budget': budget,
              'reps': reps, 'base_seed': base_seed, 'dtype': args.dtype}
    params.update(get_algorithm_params(args))

    fields_old, rows_old = read_summary(summary_path)
    best_prev = read_best_mean(rows_old, d, budget)
    is_best = float(st['mean']) < best_prev

    row = {'timestamp': datetime.now().isoformat(timespec='seconds'), **params}
    row.update({k: repr(float(st[k])) for k in STAT_FIELDS})
    append_summary(summary_path, row, fields_old, rows_old)

    print("\nFinal normalized value over {} reps: mean={:.6e} median={:.6e} std={:.6e} "
          "min={:.6e} max={:.6e} p05={:.6e} p95={:.6e}".format(
              reps, st['mean'], st['median'], st['std'], st['min'], st['max'], st['p05'], st['p95']))
    print(f"Summary appended to {summary_path}")

    if is_best:
        write_trace(trace_path, params, evals, histories)
        print(f"New best average (previous: {best_prev:.6e}) -> trace written to {trace_path}")
    else:
        print(f"Average not better than best stored ({best_prev:.6e}); trace not updated")


if __name__ == '__main__':
    main(get_arguments())