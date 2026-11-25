import argparse as ap
import csv
import json
import os
import sys
from datetime import datetime

import time

import numpy as np
import torch
from synthetic_targets import TARGETS

sys.path.append("../../../")

from algo_utils import add_algorithm_subparsers, build_algorithm, get_algorithm_params

DTYPES = {'float32': torch.float32, 'float64': torch.float64}

EPS = 1e-10   # added to numerator and denominator of the normalized gap to avoid 0/0

STAT_FIELDS = ['mean', 'median', 'std', 'sem', 'ci95_halfwidth', 'min', 'max', 'p05', 'p95']
# 'last' = evaluation-weighted mean of the per-iteration best over the last fraction of the budget
# 'best' = best value found over the entire trajectory
SCORES = ['last', 'best']
STAT_COLS = [f'{s}_{k}' for s in SCORES for k in STAT_FIELDS]
COMMON_PARAMS = ['algorithm', 'target', 'd', 'budget', 'reps', 'base_seed', 'dtype', 'last_frac',
                 'div_thresh', 'div_patience']
STATUS_COLS = ['diverged', 'n_reps_done']   # run outcome, not part of the configuration


def get_arguments():
    common = ap.ArgumentParser(add_help=False)
    common.add_argument("target", default='ackley', choices=list(TARGETS.keys()), help='Target function')
    common.add_argument("d", default=1, type=int, help="Dimension")
    common.add_argument("--reps", default=10, type=int, help='Number of repetitions')
    common.add_argument("--lam-target", default=1e-4, type=float, help='Regularization parameter of the target')
    common.add_argument("--budget", default=10000, type=int, help='Number of function evaluations')
    common.add_argument("--base-seed", default=1234, type=int, help='Base used to compute random seeds')
    common.add_argument("--dtype", default='float32', choices=list(DTYPES.keys()), help='Data type')
    common.add_argument("--device", default='cuda', choices=['cpu', 'cuda'], help='Device')
    common.add_argument("--out-dir", default="./results", type=str, help="Output directory")
    common.add_argument("--last-frac", default=0.05, type=float,
                        help='Last score = evaluation-weighted mean of the per-iteration best value '
                             'over the last fraction of the budget')
    common.add_argument("--div-thresh", default=10.0, type=float,
                        help='Diverged if the normalized local value (1 = starting point) exceeds this')
    common.add_argument("--div-patience", default=5, type=int,
                        help='Number of consecutive iterations above --div-thresh needed to declare '
                             'divergence (non-finite values always stop immediately)')

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


def read_best_mean(rows, d, budget, last_frac, col='last_mean'):
    best = float('inf')
    for row in rows:
        try:
            if _norm(row.get('diverged')) == 'True':
                continue
            if (int(row['d']) != d or int(row['budget']) != budget
                    or float(row['last_frac']) != last_frac):
                continue
            best = min(best, float(row[col]))
        except (KeyError, ValueError, TypeError):
            continue
    return best


def _norm(v):
    return '' if v is None else str(v)


def config_already_tested(rows, params):
    for row in rows:
        if all(_norm(v) == _norm(row.get(k)) for k, v in params.items()):
            return True
    return False


def append_summary(path, row, fieldnames_old, rows_old):
    fields = ['timestamp'] + COMMON_PARAMS + STATUS_COLS
    extra = [k for k in row if k not in fields and k not in STAT_COLS]
    fields = fields + extra + STAT_COLS
    for k in fieldnames_old:
        if k not in fields:
            fields.insert(len(fields) - len(STAT_COLS), k)

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


def write_trace(path, params, evals, local_histories, best_histories):
    st_local = compute_stats(local_histories, axis=0)
    st_best = compute_stats(best_histories, axis=0)
    tmp = path + ".tmp"
    with open(tmp, 'w', newline='') as fout:
        fout.write("# params: " + json.dumps(params) + "\n")
        writer = csv.writer(fout)
        writer.writerow(['iteration', 'evaluations']
                        + [f'local_{k}' for k in STAT_FIELDS]
                        + [f'running_{k}' for k in STAT_FIELDS])
        for i in range(local_histories.shape[1]):
            writer.writerow([i, int(evals[i])]
                            + [repr(float(st_local[k][i])) for k in STAT_FIELDS]
                            + [repr(float(st_best[k][i])) for k in STAT_FIELDS])
    os.replace(tmp, path)


def main(args):
    algo_name, target_name = args.algorithm, args.target
    d, reps, budget, base_seed = args.d, args.reps, args.budget, args.base_seed
    dtype, device = DTYPES[args.dtype], args.device
    last_frac = args.last_frac
    assert 0.0 < last_frac <= 1.0, "--last-frac must be in (0, 1]"
    div_thresh, div_patience = args.div_thresh, args.div_patience
    assert div_patience >= 1, "--div-patience must be >= 1"

    lam_target = args.lam_target

    out_dir = os.path.join(args.out_dir, "synthetic", target_name)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{algo_name}_summary.csv")
    trace_path = os.path.join(out_dir, f"{algo_name}_d{d}_b{budget}_f{last_frac}_best_trace.csv")

    params = {'algorithm': algo_name, 'target': target_name, 'd': d, 'budget': budget,
              'reps': reps, 'base_seed': base_seed, 'dtype': args.dtype,
              'last_frac': last_frac, 'div_thresh': div_thresh, 'div_patience': div_patience}
    params.update(get_algorithm_params(args))

    fields_old, rows_old = read_summary(summary_path)
    if config_already_tested(rows_old, params):
        print(f"Configuration already tested in {summary_path}; skipping.")
        print(json.dumps(params))
        return

    best_histories, local_histories, eval_counts = [], [], []
    rep_last, rep_best = [], []
    run_diverged = False
    for rep in range(reps):
        init_seed = 213 * base_seed + 154 + rep

        target = TARGETS[target_name](d=d, dtype=dtype, device=device, lam=lam_target, seed=init_seed)
        x0 = target.sample(1)

        ctx = {'x0': x0, 'd': d, 'rep': rep, 'base_seed': base_seed, 'device': device, 'dtype': dtype}
        opt = build_algorithm(args, ctx)

        eval_performed = 0
        f_0 = target(x0).item()
        min_f = target(target.x_star).item()
        denom = f_0 - min_f + EPS

        best_f = f_0
        n_bad = 0                                  # consecutive iterations above div_thresh
        rep_diverged = False
        best_hist, local_hist, evals = [1.0], [1.0], [0]   # index 0 = starting point
        opt_time = time.time()
        while eval_performed < budget:
            points = opt.ask()
            values = target(points)
            opt.tell(points, values)
            eval_performed += points.shape[0]

            local_best = values.min().item()      # best of this iteration only (no running min)
            best_f = min(best_f, local_best)
            best_hist.append((best_f - min_f + EPS) / denom)
            local_hist.append((local_best - min_f + EPS) / denom)
            evals.append(eval_performed)
            print(f"[rep {rep + 1}/{reps}] [{eval_performed}/{budget}] "
                  f"running best normalized: {best_hist[-1]:.6e} "
                  f"local normalized: {local_hist[-1]:.6e} local best: {local_best}")

            # Divergence check on the CURRENT iteration's value (the running best is
            # monotone and can never exceed 1, so it cannot detect divergence).
            if not np.isfinite(local_hist[-1]):
                rep_diverged = True
            elif local_hist[-1] > div_thresh:
                n_bad += 1
                rep_diverged = n_bad >= div_patience
            else:
                n_bad = 0
            if rep_diverged:
                break
        opt_time = time.time() - opt_time

        print(f"[--] Optimization time: {opt_time}")

        if rep_diverged:
            run_diverged = True
            print(f"[rep {rep + 1}/{reps}] DIVERGED after {eval_performed} evaluations "
                  f"(normalized local = {local_hist[-1]:.6e}, threshold = {div_thresh}); "
                  f"stopping this configuration")
            break

        # Windowed score: evaluation-weighted mean of the per-iteration best over the last
        # `last_frac` of the budget (each iteration weighted by the evaluations it used).
        ev = np.asarray(evals[1:])                # drop the starting point
        loc = np.asarray(local_hist[1:])
        w = np.diff(evals)                        # evaluations used by each iteration
        mask = ev > budget * (1.0 - last_frac)
        if not mask.any():
            mask[-1] = True
        rep_last.append(float(np.average(loc[mask], weights=w[mask])))
        rep_best.append(float(best_hist[-1]))     # best over the whole trajectory

        best_histories.append(best_hist)
        local_histories.append(local_hist)
        eval_counts.append(evals)

    if run_diverged:
        row = {'timestamp': datetime.now().isoformat(timespec='seconds'), **params,
               'diverged': True, 'n_reps_done': rep + 1}
        row.update({f'{s}_{k}': repr(float('nan')) for s in SCORES for k in STAT_FIELDS})
        append_summary(summary_path, row, fields_old, rows_old)
        print(f"Run marked as diverged in {summary_path}; trace not written")
        return

    n_it = min(len(h) for h in best_histories)
    best_histories = np.array([h[:n_it] for h in best_histories], dtype=np.float64)
    local_histories = np.array([h[:n_it] for h in local_histories], dtype=np.float64)
    evals = np.array(eval_counts[0][:n_it])

    stats = {'last': compute_stats(np.array(rep_last, dtype=np.float64)),
             'best': compute_stats(np.array(rep_best, dtype=np.float64))}

    best_prev = read_best_mean(rows_old, d, budget, last_frac, col='last_mean')
    is_best = float(stats['last']['mean']) < best_prev

    row = {'timestamp': datetime.now().isoformat(timespec='seconds'), **params,
           'diverged': False, 'n_reps_done': reps}
    for s in SCORES:
        row.update({f'{s}_{k}': repr(float(stats[s][k])) for k in STAT_FIELDS})
    append_summary(summary_path, row, fields_old, rows_old)

    for s in SCORES:
        st = stats[s]
        print("\n[{}] over {} reps: mean={:.6e} median={:.6e} std={:.6e} "
              "min={:.6e} max={:.6e} p05={:.6e} p95={:.6e}".format(
                  s, reps, st['mean'], st['median'], st['std'], st['min'], st['max'],
                  st['p05'], st['p95']))
    print(f"Summary appended to {summary_path}")

    if is_best:
        write_trace(trace_path, params, evals, local_histories, best_histories)
        print(f"New best last_mean (previous: {best_prev:.6e}) -> trace written to {trace_path}")
    else:
        print(f"last_mean not better than best stored ({best_prev:.6e}); trace not updated")


if __name__ == '__main__':
    main(get_arguments())