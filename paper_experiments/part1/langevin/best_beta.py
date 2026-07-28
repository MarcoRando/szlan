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

FUNCTIONS  = ['ackley', 'levy', 'rastrigin']#['ackley', 'rastrigin', 'rosenbrock']
DIMENSIONS = [10, 50, 100, 1000]



ZLABEL = r'$\frac{\min_{k \leq T} F(x_k) - \min F}{F(x_0) - \min F}$'


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log_file(filepath):
    gammas, betas, mus, stds = [], [], [], []
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
                if float(parts[2]) < 1.0 and float(parts[5]) < 1.0:
                    mus.append(float(parts[5]))
                    stds.append(float(parts[6]))
                else:
                    mus.append(1.0)
                    stds.append(0.0)#float(parts[6]))
#                mus.append(float(parts[5]))
#                stds.append(float(parts[6]))
            except ValueError:
                continue
    return (np.array(gammas), np.array(betas),
            np.array(mus),    np.array(stds))


# ---------------------------------------------------------------------------
# Reduction: for each unique value in key_arr pick the row minimising mu
# ---------------------------------------------------------------------------

def _best_curve(key_arr, other_arr, mu, std):
    unique_keys = np.unique(key_arr)
    best_mu  = np.empty(len(unique_keys))
    best_std = np.empty(len(unique_keys))
    for i, k in enumerate(unique_keys):
        mask     = key_arr == k
        idx_best = np.argmin(mu[mask])
        best_mu[i]  = mu[mask][idx_best]
        best_std[i] = std[mask][idx_best]
    return unique_keys, best_mu, best_std


# ---------------------------------------------------------------------------
# Single panel helper
# ---------------------------------------------------------------------------

def _draw_panel(ax, x, mu, std, xlabel, color, show_ylabel=False):
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylim(0, 1)
    if show_ylabel:
        ax.set_ylabel(ZLABEL, fontsize=10)
    else:
        ax.set_yticklabels([])

    ax.plot(x, mu, color=color, linewidth=1.2, marker="o", markersize=3, rasterized=True)
    ax.fill_between(x,
                    np.clip(mu - std, 0, 1),
                    np.clip(mu + std, 0, 1),
                    alpha=0.25, color=color, rasterized=True)


# ---------------------------------------------------------------------------
# Main plot: 2 rows × n_funcs cols
#   row 0 — x = gamma (stepsize), red,   best beta per gamma
#   row 1 — x = beta,             black, best gamma per beta
# ---------------------------------------------------------------------------

def plot(result_dir, dimension):
    n_funcs = len(FUNCTIONS)
    fig, axes = plt.subplots(2, n_funcs,
                             figsize=(n_funcs * 2.8, 5.5),
                             constrained_layout=True)

    for col, func in enumerate(FUNCTIONS):
        filename = f"{func}_{dimension}_tested.log"
        filepath = os.path.join(result_dir, filename)
        title    = func.replace('_', ' ').capitalize()

        axes[0, col].set_title(title, pad=4)

        if not os.path.isfile(filepath):
            print(f"[WARN] File not found: {filepath}")
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'File not found',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes,
                                    color='gray', fontsize=8)
            continue

        gamma, beta, mu, std = parse_log_file(filepath)
        mu = np.clip(mu, None, 1.0)

        # Row 0: x = beta, best gamma — black
        b_sorted, mu_b, std_b = _best_curve(beta, gamma, mu, std)
        _draw_panel(axes[0, col], b_sorted, mu_b, std_b,
                    r'$\beta$', color='#111111', show_ylabel=(col == 0))

        # Row 1: x = gamma (stepsize), best beta — red
        g_sorted, mu_g, std_g = _best_curve(gamma, beta, mu, std)
        _draw_panel(axes[1, col], g_sorted, mu_g, std_g,
                    r'$\gamma$', color='#cc0000', show_ylabel=(col == 0))

    out_path = os.path.join(result_dir, f'results_best_beta_d{dimension}.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot mu vs gamma (best beta) and mu vs beta (best gamma), '
            'confidence band from std, one subplot per function.'
        )
    )
    parser.add_argument(
        'result_dir',
        help='Directory containing <function>_<dimension>_tested.log files',
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