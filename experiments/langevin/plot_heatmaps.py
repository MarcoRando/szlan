import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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
    'pdf.fonttype':       42,   # embeds fonts properly
    'ps.fonttype':        42,
})

FUNCTIONS  = ['ackley', 'levy', 'rastrigin']#['ackley', 'rastrigin', 'rosenbrock']
#FUNCTIONS  = ['ackley', 'levy', 'rastrigin']#['ackley', 'rastrigin', 'rosenbrock']
DIMENSIONS = [10, 50, 100, 1000]


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
            except ValueError:
                continue
    return np.array(gammas), np.array(betas), np.array(mus), np.array(stds)


# ---------------------------------------------------------------------------
# Core plotting primitive
# ---------------------------------------------------------------------------

def plot_heatmap(ax, gamma, beta, mu, title,
                 show_xlabel=True, show_ylabel=True):
    if len(gamma) == 0:
        ax.set_title(title, fontsize=9)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return None

    mu = np.clip(mu, None, 1.0)
    eps = 1e-12
    log_gamma = np.log10(np.where(gamma > 0, gamma, eps))
    log_beta  = np.log10(np.where(beta  > 0, beta,  eps))

    GRID  = 150
    lg_lin = np.linspace(log_gamma.min(), log_gamma.max(), GRID)
    lb_lin = np.linspace(log_beta.min(),  log_beta.max(),  GRID)
    LG, LB = np.meshgrid(lg_lin, lb_lin)

    try:
        Z = griddata((log_gamma, log_beta), mu, (LG, LB), method='cubic')
        Z_lin = griddata((log_gamma, log_beta), mu, (LG, LB), method='linear')
        Z = np.where(np.isnan(Z), Z_lin, Z)
    except Exception:
        Z = griddata((log_gamma, log_beta), mu, (LG, LB), method='linear')

    Z = np.clip(Z, 0, 1.0)

    im = ax.imshow(
        Z, origin='lower', aspect='auto',
        extent=[lg_lin[0], lg_lin[-1], lb_lin[0], lb_lin[-1]],
        cmap='viridis', vmin=0, vmax=1, interpolation='bilinear',
    )
    ax.scatter(log_gamma, log_beta, c=mu, cmap='viridis', vmin=0, vmax=1,
               s=6, edgecolors='white', linewidths=0.2, zorder=5, alpha=0.75)

    ax.set_title(title, pad=4, fontsize=10)

    def nice_log_ticks(lo, hi, n=4):
        candidates = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if len(candidates) == 0:
            return np.linspace(lo, hi, n)
        if len(candidates) <= n:
            return candidates
        idx = np.round(np.linspace(0, len(candidates) - 1, n)).astype(int)
        return candidates[idx]

    xt = nice_log_ticks(log_gamma.min(), log_gamma.max())
    yt = nice_log_ticks(log_beta.min(),  log_beta.max())

    ax.set_xticks(xt)
    ax.set_xticklabels([f'$10^{{{int(v)}}}$' for v in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f'$10^{{{int(v)}}}$' for v in yt])

    if show_xlabel:
        ax.set_xlabel(r'$\gamma$')
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(r'$\beta$')
    else:
        ax.set_yticklabels([])

    return im


def _add_colorbar(fig, im, ax_list):
    """Add a shared colorbar aligned to the provided axes."""
    cbar = fig.colorbar(im, ax=ax_list, fraction=0.025, pad=0.03,
                        aspect=30, shrink=0.95)
    cbar.set_label(
        r'$\frac{\min_{k \leq T} F(x_k) - \min F}{F(x_0) - \min F}$',
        labelpad=4, fontsize=10,
    )
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7, direction='in', width=0.6)
    return cbar


# ---------------------------------------------------------------------------
# Layout: combined
#   no dimension  → n_funcs × 3  grid
#   with dimension → 1 × n_funcs grid (one col per function)
# ---------------------------------------------------------------------------

def plot_combined(result_dir, dimension=None):
    n_funcs = len(FUNCTIONS)
    dims = [dimension] if dimension is not None else DIMENSIONS

    if dimension is not None:
        n_rows, n_cols = 1, n_funcs
        figsize = (n_funcs * 2.3, 2.5)
        suffix = f'_d{dimension}'
    else:
        n_rows, n_cols = n_funcs, 3
        figsize = (6.7, n_funcs * 2.1)
        suffix = ''

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             constrained_layout=True)
    # Normalise axes to always be 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    last_im = None

    if dimension is not None:
        # Single row: columns = functions
        for col, func in enumerate(FUNCTIONS):
            ax = axes[0, col]
            filename = f"{func}_{dimension}_tested.log"
            filepath = os.path.join(result_dir, filename)
            title = func.replace('_', ' ').capitalize()

            if not os.path.isfile(filepath):
                print(f"[WARN] File not found: {filepath}")
                ax.set_title(title, pad=4)
                ax.text(0.5, 0.5, 'File not found', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=7)
                continue

            gamma, beta, mu, _ = parse_log_file(filepath)
            im = plot_heatmap(ax, gamma, beta, mu, title=title,
                              show_xlabel=True, show_ylabel=(col == 0))
            if im is not None:
                last_im = im

#        fig.suptitle(f'$d = {dimension}$', fontsize=9, fontweight='bold')

    else:
        # Rows = functions, cols = dimensions
        for row, func in enumerate(FUNCTIONS):
            for col, dim in enumerate(dims):
                ax = axes[row, col]
                filename = f"{func}_{dim}_tested.log"
                filepath = os.path.join(result_dir, filename)

                if not os.path.isfile(filepath):
                    print(f"[WARN] File not found: {filepath}")
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center',
                            transform=ax.transAxes, color='gray', fontsize=7)
                    continue

                gamma, beta, mu, _ = parse_log_file(filepath)
                im = plot_heatmap(
                    ax, gamma, beta, mu,
                    title='',#f'$d = {dim}$' if row == 0 else '',
                    show_xlabel=(row == n_funcs - 1),
                    show_ylabel=(col == 0),
                )
                if im is not None:
                    last_im = im

            # Row ylabel carries the function name
            axes[row, 0].set_ylabel(
                rf'$\beta$  —  {func.replace("_", " ").capitalize()}',
                labelpad=6,
            )

    if last_im is not None:
        _add_colorbar(fig, last_im, axes.tolist()[0])

    out_path = os.path.join(result_dir, f'results_heatmap_combined{suffix}.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Layout: separated
#   no dimension  → one 1×3 figure per function (cols = dims)
#   with dimension → one single-panel figure per function
# ---------------------------------------------------------------------------

def plot_separated(result_dir, dimension=None):
    dims = [dimension] if dimension is not None else DIMENSIONS
    n_cols = len(dims)
    suffix = f'_d{dimension}' if dimension is not None else ''

    for func in FUNCTIONS:
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(n_cols * 2.3, 2.3),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)   # keep iterable when n_cols == 1
        last_im = None

        for col, dim in enumerate(dims):
            ax = axes[col]
            filename = f"{func}_{dim}_tested.log"
            filepath = os.path.join(result_dir, filename)
            title = f'$d = {dim}$'

            if not os.path.isfile(filepath):
                print(f"[WARN] File not found: {filepath}")
                ax.set_title(title, pad=4)
                ax.text(0.5, 0.5, 'File not found', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=7)
                continue

            gamma, beta, mu, _ = parse_log_file(filepath)
            im = plot_heatmap(ax, gamma, beta, mu, title=title,
                              show_xlabel=True, show_ylabel=(col == 0))
            if im is not None:
                last_im = im

        fig.suptitle(func.replace('_', ' ').capitalize(), x=0.02, ha='left',
                     fontsize=9, fontweight='bold')

        if last_im is not None:
            _add_colorbar(fig, last_im, axes.tolist())

        out_path = os.path.join(result_dir, f'results_heatmap_{func}{suffix}.pdf')
        fig.savefig(out_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Saved → {out_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot heatmaps of μ-value (publication quality, PDF output).'
    )
    parser.add_argument(
        'result_dir',
        help='Directory containing <function>_<dimension>_tested.log files',
    )
    parser.add_argument(
        '--layout',
        choices=['combined', 'separated'],
        default='combined',
        help='combined (default): single grid figure  |  separated: one figure per function',
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=None,
        choices=DIMENSIONS,
        metavar='{' + ','.join(str(d) for d in DIMENSIONS) + '}',
        help=(
            'Plot only this dimension. '
            'combined → 1×N_funcs figure; '
            'separated → one single-panel figure per function.'
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"[ERROR] Directory not found: {args.result_dir}")
        exit(1)

    if args.layout == 'separated':
        plot_separated(args.result_dir, dimension=args.dimension)
    else:
        plot_combined(args.result_dir, dimension=args.dimension)


if __name__ == '__main__':
    main()