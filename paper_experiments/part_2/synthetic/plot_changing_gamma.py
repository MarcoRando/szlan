#!/usr/bin/env python3
"""
SZLAN Hyperparameter Analysis & Plotting Script

Generates two figures based on grid search summary results (szlan_summary.csv):
1. Sensitivity Curves (2x3 subplots):
   - Row 1 (horizontal): x-axis is gamma, y-axis is the mean (+/- std) for the best beta
     (lowest mean) at s = 1, connected for each value of l, for takagi, yang, zigzag.
     Normalized values exceeding 1.0 are clipped to 1.0.
   - Row 2 (horizontal): Identical to Row 1, but with x-axis as beta and selecting the best gamma.
2. 2D Performance Landscapes as Heatmaps (1x3 subplots):
   - Horizontal heatmaps for takagi, yang, zigzag.
   - x-axis: gamma, y-axis: beta.
   - Each cell represents the mean performance for that (gamma, beta) choice at s = 1,
     optimized over l (best l), clipped to 1.0 if exceeding 1.0.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Mapping of target nicknames to canonical names and display titles
TARGET_INFO = {
    'takagi': {
        'canonical': 'takagi_l1',
        'title': r'Takagi ($L_1$)',
    },
    'takagi_l1': {
        'canonical': 'takagi_l1',
        'title': r'Takagi ($L_1$)',
    },
    'yang': {
        'canonical': 'yang_n2',
        'title': r'Yang ($N_2$)',
    },
    'yang_n2': {
        'canonical': 'yang_n2',
        'title': r'Yang ($N_2$)',
    },
    'zigzag': {
        'canonical': 'zigzag_l1',
        'title': r'Zigzag ($L_1$)',
    },
    'zigzag_l1': {
        'canonical': 'zigzag_l1',
        'title': r'Zigzag ($L_1$)',
    },
    'ackley': {
        'canonical': 'ackley',
        'title': 'Ackley',
    },
    'rastrigin': {
        'canonical': 'rastrigin',
        'title': 'Rastrigin',
    },
}

DEFAULT_TARGETS = ['ackley', 'rastrigin', 'zigzag']# ['takagi', 'yang', 'zigzag']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '8']


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot SZLAN parameter sensitivity curves and 2D heatmaps."
    )
    parser.add_argument(
        "--d",
        type=int,
        default=10,
        help="Problem dimension d (default: 10)",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=1,
        help="Subsampling dimension parameter s (default: 1)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="List of target functions (default: takagi yang zigzag)",
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="Path to results/grid directory (default: auto-detected)",
    )
    parser.add_argument(
        "--out-curves",
        type=str,
        default=None,
        help="Output path for the sensitivity curves plot (default: szlan_curves_d{d}.pdf)",
    )
    parser.add_argument(
        "--out-landscape",
        type=str,
        default=None,
        help="Output path for the 2D heatmap plot (default: szlan_landscape_d{d}.pdf)",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=1.0,
        help="Ceiling value for normalized performance (default: 1.0)",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable clipping of normalized values exceeding clip-max",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear y-axis scale for sensitivity curves instead of log scale",
    )
    parser.add_argument(
        "--best-l-by",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Metric used to select the best l for each (gamma, beta) in heatmaps (default: mean)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis_r",
        help="Colormap for heatmaps (default: viridis_r, lower/better values are brighter)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for figures (default: 300)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable automatic export of accompanying PNG files",
    )
    return parser.parse_args()


def find_grid_dir(explicit_dir=None):
    """Locate results/grid directory robustly across working directories."""
    candidates = []
    if explicit_dir is not None:
        candidates.append(explicit_dir)

    # Current working directory
    candidates.append("./results/grid")
    # Relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "results", "grid"))
    # Relative to repository root
    candidates.append(os.path.join(script_dir, "..", "..", "..", "paper_experiments", "part_2", "synthetic", "results", "grid"))

    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        f"Could not locate results/grid directory. Checked paths:\n" +
        "\n".join(f"  - {c}" for c in candidates) +
        "\nPlease specify --grid-dir explicitly."
    )


def load_summary(grid_dir, target_key, d):
    """Load szlan_summary.csv for a given target function and dimension d."""
    info = TARGET_INFO.get(target_key.lower(), {
        'canonical': target_key,
        'title': target_key.replace('_', ' ').capitalize()
    })
    target_name = info['canonical']

    expected_path = os.path.join(
        grid_dir,
        f"szlan_{target_name}_d{d}",
        "synthetic",
        target_name,
        "szlan_summary.csv"
    )

    if not os.path.exists(expected_path):
        # Walk directory in case folder naming differs slightly
        matches = []
        for root, _, files in os.walk(grid_dir):
            if "szlan_summary.csv" in files and f"d{d}" in root and target_name in root:
                matches.append(os.path.join(root, "szlan_summary.csv"))
        if matches:
            expected_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Summary CSV not found for target '{target_key}' (canonical: '{target_name}') at d={d}.\n"
                f"Searched: {expected_path}"
            )

    df = pd.read_csv(expected_path)
    return df, info['title']


def format_gamma(val):
    """Format gamma value cleanly for tick labels."""
    return f"{val:.1e}"


def format_beta(val):
    """Format beta value cleanly for tick labels."""
    if val >= 10.0:
        return f"{val:.0f}"
    elif val >= 1.0:
        return f"{val:.1f}"
    else:
        return f"{val:.2g}"


def plot_sensitivity_curves(data_by_target, d, s, clip_max=1.0, clip=True, linear_y=False, out_path="szlan_curves.pdf", dpi=300, save_png=True):
    """
    Generate the 2x3 sensitivity curves figure:
    - Row 0 (Top): x-axis is gamma, y-axis is mean (+/- std) for the best beta.
    - Row 1 (Bottom): x-axis is beta, y-axis is mean (+/- std) for the best gamma.
    Values exceeding clip_max are clipped to clip_max.
    """
    n_targets = len(data_by_target)
    fig, axes = plt.subplots(2, n_targets, figsize=(5.3 * n_targets, 9.2), squeeze=False)

    # Gather all l values across all targets to guarantee consistent styling
    all_l = set()
    for _, df, _ in data_by_target:
        df_s = df[df['s'] == s]
        all_l.update(df_s['l'].unique())
    sorted_l = sorted(all_l)

    # Perceptually distinct colors and markers for each l
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_l), 10)))
    l_to_color = {l_val: colors[i % len(colors)] for i, l_val in enumerate(sorted_l)}
    l_to_marker = {l_val: MARKERS[i % len(MARKERS)] for i, l_val in enumerate(sorted_l)}

    legend_handles = {}

    for col_idx, (target_key, df, title) in enumerate(data_by_target):
        df_s = df[df['s'] == s]
        if df_s.empty:
            raise ValueError(
                f"Target '{target_key}' has no data with s={s}. Available s values: {sorted(df['s'].unique())}"
            )

        unique_ls = sorted(df_s['l'].unique())

        # ====================================================
        # Row 0: Varying gamma on x-axis (Best beta for each l)
        # ====================================================
        ax_top = axes[0, col_idx]
        for l_val in unique_ls:
            df_l = df_s[df_s['l'] == l_val]
            gammas = []
            means = []
            lowers = []
            uppers = []

            for gamma_val, group_g in df_l.groupby('gamma'):
                best_idx = group_g['mean'].idxmin()
                best_row = group_g.loc[best_idx]
                mu = best_row['mean']
                sigma = best_row['std']

                if clip and mu >= clip_max:
                    mu_val = clip_max
                    lcb = clip_max
                    ucb = clip_max
                else:
                    mu_val = mu
                    lcb = max(mu - sigma, 1e-12 if not linear_y else 0.0)
                    ucb = min(mu + sigma, clip_max) if clip else (mu + sigma)

                gammas.append(gamma_val)
                means.append(mu_val)
                lowers.append(lcb)
                uppers.append(ucb)

            gammas = np.array(gammas)
            means = np.array(means)
            lowers = np.array(lowers)
            uppers = np.array(uppers)

            # Sort ascending by gamma
            sort_order = np.argsort(gammas)
            gammas = gammas[sort_order]
            means = means[sort_order]
            lowers = lowers[sort_order]
            uppers = uppers[sort_order]

            lower_err = np.maximum(0, means - lowers)
            upper_err = np.maximum(0, uppers - means)

            line_container = ax_top.errorbar(
                gammas,
                means,
                yerr=[lower_err, upper_err],
                marker=l_to_marker[l_val],
                color=l_to_color[l_val],
                linewidth=1.8,
                markersize=6,
                capsize=3.5,
                capthick=1.0,
                label=f"$l = {l_val}$",
            )
            if l_val not in legend_handles:
                legend_handles[l_val] = line_container

        ax_top.set_xscale('log')
        if not linear_y:
            ax_top.set_yscale('log')
        ax_top.set_title(f"{title}\n(Best $\\beta$, varying $\\gamma$)", fontsize=13, fontweight='bold', pad=8)
        ax_top.set_xlabel(r"$\gamma$", fontsize=13)
        if col_idx == 0:
            ax_top.set_ylabel(r"Mean normalized value ($\pm$ std)", fontsize=11)
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        # ====================================================
        # Row 1: Varying beta on x-axis (Best gamma for each l)
        # ====================================================
        ax_bot = axes[1, col_idx]
        for l_val in unique_ls:
            df_l = df_s[df_s['l'] == l_val]
            betas = []
            means = []
            lowers = []
            uppers = []

            for beta_val, group_b in df_l.groupby('beta'):
                best_idx = group_b['mean'].idxmin()
                best_row = group_b.loc[best_idx]
                mu = best_row['mean']
                sigma = best_row['std']

                if clip and mu >= clip_max:
                    mu_val = clip_max
                    lcb = clip_max
                    ucb = clip_max
                else:
                    mu_val = mu
                    lcb = max(mu - sigma, 1e-12 if not linear_y else 0.0)
                    ucb = min(mu + sigma, clip_max) if clip else (mu + sigma)

                betas.append(beta_val)
                means.append(mu_val)
                lowers.append(lcb)
                uppers.append(ucb)

            betas = np.array(betas)
            means = np.array(means)
            lowers = np.array(lowers)
            uppers = np.array(uppers)

            sort_order = np.argsort(betas)
            betas = betas[sort_order]
            means = means[sort_order]
            lowers = lowers[sort_order]
            uppers = uppers[sort_order]

            lower_err = np.maximum(0, means - lowers)
            upper_err = np.maximum(0, uppers - means)

            ax_bot.errorbar(
                betas,
                means,
                yerr=[lower_err, upper_err],
                marker=l_to_marker[l_val],
                color=l_to_color[l_val],
                linewidth=1.8,
                markersize=6,
                capsize=3.5,
                capthick=1.0,
                label=f"$l = {l_val}$",
            )

        ax_bot.set_xscale('log')
        if not linear_y:
            ax_bot.set_yscale('log')
        ax_bot.set_title(f"{title}\n(Best $\\gamma$, varying $\\beta$)", fontsize=13, fontweight='bold', pad=8)
        ax_bot.set_xlabel(r"$\beta$", fontsize=13)
        if col_idx == 0:
            ax_bot.set_ylabel(r"Mean normalized value ($\pm$ std)", fontsize=11)
        ax_bot.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Single unified horizontal legend at the top
    handles = [legend_handles[l_val] for l_val in sorted_l if l_val in legend_handles]
    labels = [f"$l = {l_val}$" for l_val in sorted_l if l_val in legend_handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(labels),
        fontsize=11,
        frameon=True,
        edgecolor="gray",
    )

    clip_str = f", clipped $\\leq {clip_max}$" if clip else ""
    fig.suptitle(
        f"SZLAN Hyperparameter Sensitivity (Mean $\\pm$ Std, $d = {d},\\; s = {s}${clip_str})",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save files
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[✓] Saved sensitivity curves plot to: {out_path}")

    if save_png and not out_path.endswith(".png"):
        png_path = os.path.splitext(out_path)[0] + ".png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"[✓] Saved sensitivity curves PNG to:  {png_path}")

    plt.close(fig)


def plot_gamma_beta_heatmaps(data_by_target, d, s, clip_max=1.0, clip=True, best_l_by="mean", cmap="viridis_r", out_path="szlan_landscape.pdf", dpi=300, save_png=True):
    """
    Generate 1x3 2D landscape heatmaps:
    - x-axis: gamma (columns), y-axis: beta (rows)
    - Each cell shows the mean performance for that (gamma, beta) at best l and s=1.
    - Values exceeding clip_max are clipped to clip_max.
    - Unevaluated grid points are displayed in neutral light gray.
    """
    n_targets = len(data_by_target)
    fig, axes = plt.subplots(1, n_targets, figsize=(6.2 * n_targets, 5.8), squeeze=False)

    # Base colormap with masked/NaN color for unevaluated entries
    base_cmap = plt.get_cmap(cmap).copy()
    base_cmap.set_bad(color='#d3d3d3')  # Light gray

    for col_idx, (target_key, df, title) in enumerate(data_by_target):
        df_s = df[df['s'] == s]
        if df_s.empty:
            raise ValueError(
                f"Target '{target_key}' has no data with s={s}. Available s values: {sorted(df['s'].unique())}"
            )

        unique_gammas = sorted(df_s['gamma'].unique())
        unique_betas = sorted(df_s['beta'].unique())

        # Select best l row for each (gamma, beta) combination
        idx_best_l = df_s.groupby(['gamma', 'beta'])[best_l_by].idxmin()
        best_df = df_s.loc[idx_best_l].copy()

        # Build 2D pivot table (index=beta, columns=gamma)
        pivot = best_df.pivot(index='beta', columns='gamma', values='mean')
        # Reindex to full product grid so missing combinations appear as NaN
        pivot = pivot.reindex(index=unique_betas, columns=unique_gammas)

        if clip:
            pivot = pivot.clip(upper=clip_max)

        matrix = pivot.values  # shape: (n_betas, n_gammas)
        ax = axes[0, col_idx]

        # Determine color normalization
        vmin = np.nanmin(matrix)
        vmax = clip_max if clip else np.nanmax(matrix)

        if vmin > 0 and (vmax / vmin) > 10:
            norm = mcolors.LogNorm(vmin=max(vmin, 1e-4), vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(
            matrix,
            origin='lower',
            aspect='auto',
            cmap=base_cmap,
            norm=norm,
            interpolation='nearest',
        )

        # Highlight the global minimum cell with a star marker
        valid_matrix = np.where(np.isnan(matrix), np.inf, matrix)
        min_flat_idx = np.argmin(valid_matrix)
        min_beta_idx, min_gamma_idx = np.unravel_index(min_flat_idx, matrix.shape)
        opt_gamma = unique_gammas[min_gamma_idx]
        opt_beta = unique_betas[min_beta_idx]
        opt_val = matrix[min_beta_idx, min_gamma_idx]

        ax.plot(
            min_gamma_idx,
            min_beta_idx,
            marker='*',
            color='red',
            markersize=13,
            markeredgecolor='black',
            markeredgewidth=0.8,
            linestyle='none',
            label=f"Min: {opt_val:.3g}\n($\\gamma={opt_gamma:.1e}, \\beta={opt_beta:.2g}$)",
        )

        # X-axis ticks (gamma)
        ax.set_xticks(range(len(unique_gammas)))
        ax.set_xticklabels([format_gamma(g) for g in unique_gammas], rotation=45, ha='right', fontsize=9)
        ax.set_xlabel(r"$\gamma$", fontsize=13)

        # Y-axis ticks (beta)
        # If there are many beta values, step ticks to avoid crowding
        step = 2 if len(unique_betas) > 12 else 1
        y_indices = list(range(0, len(unique_betas), step))
        ax.set_yticks(y_indices)
        ax.set_yticklabels([format_beta(unique_betas[i]) for i in y_indices], fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(r"$\beta$", fontsize=13)

        ax.set_title(f"{title}\n(Best $l$ selected by {best_l_by})", fontsize=13, fontweight='bold', pad=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar_label = r"Mean normalized value (clipped $\leq 1$)" if clip else "Mean normalized value"
        cbar.set_label(cbar_label, fontsize=10)

        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    clip_str = f", clipped $\\leq {clip_max}$" if clip else ""
    fig.suptitle(
        f"SZLAN Mean Performance Heatmaps vs $(\\gamma, \\beta)$ ($d = {d},\\; s = {s}${clip_str})",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[✓] Saved landscape heatmap to: {out_path}")

    if save_png and not out_path.endswith(".png"):
        png_path = os.path.splitext(out_path)[0] + ".png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"[✓] Saved landscape heatmap PNG to:  {png_path}")

    plt.close(fig)


def main():
    args = parse_arguments()

    grid_dir = find_grid_dir(args.grid_dir)
    print(f"[*] Results grid root: {grid_dir}")
    print(f"[*] Dimension d={args.d}, parameter s={args.s}")
    print(f"[*] Target functions: {args.targets}")
    print(f"[*] Clipping enabled: {not args.no_clip} (ceiling: {args.clip_max})")

    # Load data for each requested target function
    data_by_target = []
    for target in args.targets:
        df, title = load_summary(grid_dir, target, args.d)
        data_by_target.append((target, df, title))
        print(f"[+] Loaded {target}: {len(df)} total rows")

    # Determine output file locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_curves = args.out_curves or os.path.join(script_dir, f"szlan_curves_d{args.d}.pdf")
    out_landscape = args.out_landscape or os.path.join(script_dir, f"szlan_landscape_d{args.d}.pdf")

    # 1. Sensitivity curves (varying gamma top, varying beta bottom) with mean +/- std
    print("\n--- Generating Sensitivity Curves Plot (2x3, Mean ± Std) ---")
    plot_sensitivity_curves(
        data_by_target,
        d=args.d,
        s=args.s,
        clip_max=args.clip_max,
        clip=not args.no_clip,
        linear_y=args.linear_y,
        out_path=out_curves,
        dpi=args.dpi,
        save_png=not args.no_png,
    )

    # 2. 2D Gamma vs Beta Heatmaps (1x3)
    print("\n--- Generating 2D (Gamma, Beta) Performance Heatmaps (1x3) ---")
    plot_gamma_beta_heatmaps(
        data_by_target,
        d=args.d,
        s=args.s,
        clip_max=args.clip_max,
        clip=not args.no_clip,
        best_l_by=args.best_l_by,
        cmap=args.cmap,
        out_path=out_landscape,
        dpi=args.dpi,
        save_png=not args.no_png,
    )

    print("\n[✓] Done! All figures generated successfully.")


if __name__ == "__main__":
    main()
