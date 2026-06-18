import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
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
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

FUNCTIONS  = ['ackley', 'levy', 'rastrigin']#['ackley', 'rastrigin', 'rosenbrock']#, 'least_squares']
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
                if float(parts[2]) < 1 and float(parts[5]):
                    mus.append(float(parts[5]))
                    stds.append(float(parts[6]))
                else:
                    mus.append(1.0)
                    stds.append(1e-20)
            except ValueError:
                continue
    return np.array(gammas), np.array(betas), np.array(mus), np.array(stds)


# ---------------------------------------------------------------------------
# Core plotting primitive
# ---------------------------------------------------------------------------

def plot_3d_surface(ax, gamma, beta, mu, title,
                    show_xlabel=True, show_ylabel=True, show_mu=True):
    if len(gamma) == 0:
        ax.set_title(title, pad=4, fontsize=12)
        return None
    mu = np.log(mu)
#    mu = np.clip(mu, None, 1.0)
    eps = 1e-12
    
    print("GAMMA: ",gamma)
    print("BETA: ",beta)
    LG, LB = np.meshgrid(gamma, beta)
    Z = griddata((gamma, beta), mu, (LG, LB), method='linear')
    
    log_gamma = np.log10(np.where(gamma > 0, gamma, eps))
    log_beta  = np.log10(np.where(beta  > 0, beta,  eps))

    # X axis → gamma, Y axis → beta
    lg_lin = np.linspace(log_gamma.min(), log_gamma.max(), 60)
    lb_lin = np.linspace(log_beta.min(),  log_beta.max(),  60)
    LG, LB = np.meshgrid(lg_lin, lb_lin)

    try:
        Z = griddata((log_gamma, log_beta), mu, (LG, LB), method='cubic')
        Z_lin = griddata((log_gamma, log_beta), mu, (LG, LB), method='linear')
        Z = np.where(np.isnan(Z), Z_lin, Z)
    except Exception:
        Z = griddata((log_gamma, log_beta), mu, (LG, LB), method='linear')

#    Z = np.clip(Z, 0, 1.0)

    surf = ax.plot_surface(
        LG, LB, Z,
        cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.90,
 #       vmin=0, vmax=1,
    )
    ax.scatter(log_gamma, log_beta, mu,
               color='#cc2222', s=6, zorder=5, alpha=0.65, depthshade=True)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor('#f5f5f5')
        pane.set_edgecolor('#cccccc')
        pane.set_alpha(0.4)
    ax.grid(True, color='#cccccc', linewidth=0.3, linestyle=':')

    ax.set_title(title, pad=4, fontsize=12)

    def nice_log_ticks(lo, hi, n=3):
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
    ax.set_xticklabels([f'$10^{{{int(v)}}}$' for v in xt], fontsize=6)
    ax.set_yticks(yt)
    ax.set_yticklabels([f'$10^{{{int(v)}}}$' for v in yt], fontsize=6)

 #   ax.set_zlim(0, 1)
 #   ax.set_zticks([0.0, 0.5, 1.0])
 #   ax.set_zticklabels(['0', '0.5', '1'], fontsize=6)
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    if show_xlabel:
        ax.set_xlabel(r'$\gamma$', labelpad=4, fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r'$\beta$', labelpad=4, fontsize=8)
    # if show_mu:
    # ax.set_zlabel(
    #     r'$\frac{\min_{k \leq T} F(x_k) - \min F}{F(x_0) - \min F}$',
    #     labelpad=4, fontsize=12,
    # )

    ax.tick_params(labelsize=10, pad=1)
    return surf


# ---------------------------------------------------------------------------
# Shared horizontal colorbar helper
# ---------------------------------------------------------------------------

def _add_colorbar_horizontal(fig, surf, bottom_gap=0.06):
    """Add a horizontal colorbar centred below all subplots."""
    cbar_ax = fig.add_axes([0.25, bottom_gap, 0.50, 0.025])
    cbar = fig.colorbar(surf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(
        r'$\log \frac{\min_{k \leq T} F(x_k) - \min F}{F(x_0) - \min F}$',
        labelpad=4, fontsize=14,
    )
#    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7, direction='in', width=0.6)


# ---------------------------------------------------------------------------
# Layout: combined
#   no dimension  → n_funcs × 3  grid
#   with dimension → 1 × n_funcs grid (one col per function)
# ---------------------------------------------------------------------------

def plot_combined(result_dir, dimension=None):
    n_funcs = len(FUNCTIONS)

    if dimension is not None:
        figsize = (12, 5)
        suffix  = f'_d{dimension}'
    else:
        figsize = (6.7, n_funcs * 1.75 + 0.8)   # extra height for colorbar
        suffix  = ''

    fig = plt.figure(figsize=figsize)
    # Reserve bottom margin for the horizontal colorbar
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.18)

    last_surf = None

    if dimension is not None:
        for col, func in enumerate(FUNCTIONS):
            ax = fig.add_subplot(1, n_funcs, col + 1, projection='3d')
            filename = f"{func}_{dimension}_tested.log"
            filepath = os.path.join(result_dir, filename)
            title = func.replace('_', ' ').capitalize()

            if not os.path.isfile(filepath):
                print(f"[WARN] File not found: {filepath}")
                ax.set_title(f'{title} — not found', fontsize=7)
                continue

            gamma, beta, mu, _ = parse_log_file(filepath)

            surf = plot_3d_surface(
                ax, gamma, beta, mu, title,
                show_xlabel=True,
                show_ylabel=True,
                show_mu=(col == len(FUNCTIONS) - 1),
            )
            if surf is not None:
                last_surf = surf

    else:
        for row, func in enumerate(FUNCTIONS):
            for col, dim in enumerate(DIMENSIONS):
                ax = fig.add_subplot(n_funcs, 3, row * 3 + col + 1, projection='3d')
                filename = f"{func}_{dim}_tested.log"
                filepath = os.path.join(result_dir, filename)
                title = f'$d = {dim}$' if row == 0 else ''

                if not os.path.isfile(filepath):
                    print(f"[WARN] File not found: {filepath}")
                    ax.set_title(f'$d={dim}$ — not found', fontsize=7)
                    continue

                gamma, beta, mu, _ = parse_log_file(filepath)
                surf = plot_3d_surface(
                    ax, gamma, beta, mu, title,
                    show_xlabel=(row == n_funcs - 1),
                    show_ylabel=(col == 0),
                    show_mu=(row == n_funcs - 1 and col == 2),
                )
                if surf is not None:
                    last_surf = surf

            y_pos = 0.18 + (n_funcs - row - 0.5) / n_funcs * (0.94 - 0.18)
            fig.text(0.005, y_pos, func.replace('_', ' ').capitalize(),
                     fontsize=8, fontweight='bold', va='center', rotation=90)

    if last_surf is not None:
        _add_colorbar_horizontal(fig, last_surf, bottom_gap=0.05)

    out_path = os.path.join(result_dir, f'results_3d_combined{suffix}.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Layout: separated
#   no dimension  → one figure per function (1 × 3 dims)
#   with dimension → one figure per function (single subplot)
# ---------------------------------------------------------------------------

def plot_separated(result_dir, dimension=None):
    for func in FUNCTIONS:
        dims   = [dimension] if dimension is not None else DIMENSIONS
        n_cols = len(dims)
        suffix = f'_d{dimension}' if dimension is not None else ''

        fig = plt.figure(figsize=(n_cols * 2.6, 3.4))   # extra height for colorbar
        fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.22,
                            wspace=0.35)
        fig.suptitle(func.replace('_', ' ').capitalize(), x=0.02, ha='left',
                     fontsize=9, fontweight='bold')

        last_surf = None
        for col, dim in enumerate(dims):
            ax = fig.add_subplot(1, n_cols, col + 1, projection='3d')
            filename = f"{func}_{dim}_tested.log"
            filepath = os.path.join(result_dir, filename)
            title = f'$d = {dim}$'

            if not os.path.isfile(filepath):
                print(f"[WARN] File not found: {filepath}")
                ax.set_title(f'{title} — not found', fontsize=7)
                continue

            gamma, beta, mu, _ = parse_log_file(filepath)
            surf = plot_3d_surface(
                ax, gamma, beta, mu, title,
                show_xlabel=True,
                show_ylabel=True,
                show_mu=(col == n_cols - 1),
            )
            if surf is not None:
                last_surf = surf

        if last_surf is not None:
            _add_colorbar_horizontal(fig, last_surf, bottom_gap=0.05)

        out_path = os.path.join(result_dir, f'results_3d_{func}{suffix}.pdf')
        fig.savefig(out_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Saved → {out_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot 3-D μ-value surfaces (publication quality, PDF output).'
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
            'separated → one single-subplot figure per function.'
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