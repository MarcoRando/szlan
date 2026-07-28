import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Publication style ---
mpl.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# FUNCTIONS  = ['ackley', 'rastrigin', 'rosenbrock']
# DIMENSIONS = [10, 50, 100]

FUNCTIONS  = ['ackley', 'levy', 'rastrigin']#['ackley', 'rastrigin', 'rosenbrock']#, 'least_squares']
DIMENSIONS = [10, 50, 100, 1000]


TOP_N      = 5


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log_file(filepath):
    """Returns numeric arrays plus the exact raw strings from the file."""
    gammas, betas, mus, stds = [], [], [], []
    gamma_strs, beta_strs    = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                gammas.append(float(parts[0]))
                betas.append(float(parts[1]))
                mus.append(float(parts[2]))
                stds.append(float(parts[3]))
                gamma_strs.append(parts[0].strip())
                beta_strs.append(parts[1].strip())
            except ValueError:
                continue
    return (np.array(gammas), np.array(betas),
            np.array(mus),    np.array(stds),
            gamma_strs,       beta_strs)


def parse_trace_file(filepath):
    means, stds = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:


                # if float(parts[0]) < 1.0 and float(parts[1]) < 1.0:
                #     mus.append(float(parts[0]))
                #     stds.append(float(parts[1]))
                # else:
                #     mus.append(1.0)
                #     stds.append(0.0)#float(parts[6]))

                means.append(float(parts[0]))
                stds.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Select top-N beta values: for each unique beta pick best gamma (min mu),
# then rank betas by that mu and return the top N (gamma_str, beta_str, mu).
# ---------------------------------------------------------------------------

def top_n_beta_pairs(gamma, beta, mu, gamma_strs, beta_strs, n=TOP_N):
    unique_betas = np.unique(beta)
    best_mu      = np.empty(len(unique_betas))
    best_g_str   = []
    best_b_str   = []

    for i, b in enumerate(unique_betas):
        mask     = beta == b
        idx_best = np.argmin(mu[mask])
        best_mu[i] = mu[mask][idx_best]
        # Retrieve the exact raw string for the winning row
        row_indices = np.where(mask)[0]
        winning_row = row_indices[idx_best]
        best_g_str.append(gamma_strs[winning_row])
        best_b_str.append(beta_strs[winning_row])

    order = np.argsort(best_mu)[:n]
    return [(best_g_str[i], best_b_str[i], best_mu[i]) for i in order]


def _strip(s):
    """Remove trailing zeros after decimal point for legend readability."""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(result_dir, dimension):
    traces_dir = os.path.join(result_dir, 'traces')
    n_funcs    = len(FUNCTIONS)

    cmap   = plt.cm.tab10
    colors = [cmap(i) for i in range(TOP_N)]

    fig, axes = plt.subplots(1, n_funcs,
                             figsize=(n_funcs * 3.2, 3.2),
                             constrained_layout=True)

    for ax, func in zip(axes, FUNCTIONS):
        log_path = os.path.join(result_dir, f"{func}_{dimension}_tested.log")
        ax.set_title(func.replace('_', ' ').capitalize(), pad=4)
        ax.set_xlabel('Iteration')
        if ax is axes[0]:
            ax.set_ylabel('Function value')

        if not os.path.isfile(log_path):
            print(f"[WARN] Log not found: {log_path}")
            ax.text(0.5, 0.5, 'Log not found', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=8)
            continue

        gamma_arr, beta_arr, mu_arr, _, gamma_strs, beta_strs = \
            parse_log_file(log_path)

        pairs = top_n_beta_pairs(gamma_arr, beta_arr, mu_arr,
                                 gamma_strs, beta_strs, n=TOP_N)

        for rank, (g_str, b_str, mu_val) in enumerate(pairs):
            # Exact filename as stored on disk
            trace_name = f"{func}_{dimension}_{g_str}_{b_str}.csv"
            trace_path = os.path.join(traces_dir, trace_name)

            if not os.path.isfile(trace_path):
                print(f"[WARN] Trace not found: {trace_path}")
                continue

            mean, std = parse_trace_file(trace_path)
            iters     = np.arange(len(mean))
            color     = colors[rank]

            # Stripped values only in the legend
            ratio = float(g_str) / float(b_str)
            label = rf'$\beta={_strip(b_str)},\,\gamma={_strip(g_str)},\,\gamma/\beta={ratio:.2g}$'
            ax.plot(iters, mean, color=color, linewidth=1.0, label=label, rasterized=True)
            ax.fill_between(iters,
                            mean - std,
                            mean + std,
                            alpha=0.20, color=color, rasterized=True)

        ax.legend(fontsize=6, framealpha=0.7, loc='upper right')

    out_path = os.path.join(result_dir, f'results_traces_d{dimension}.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot convergence traces for the top-5 beta values '
            '(selected by best gamma) for each function.'
        )
    )
    parser.add_argument(
        'result_dir',
        help='Directory containing <function>_<dimension>_tested.log and traces/ subdir',
    )
    parser.add_argument(
        'dimension',
        type=int,
        choices=DIMENSIONS,
        metavar='{' + ','.join(str(d) for d in DIMENSIONS) + '}',
        help='Dimension to plot',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"[ERROR] Directory not found: {args.result_dir}")
        exit(1)

    plot(args.result_dir, args.dimension)


if __name__ == '__main__':
    main()