import argparse as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    p = ap.ArgumentParser(description="Plot median and percentiles vs gamma for different l")
    p.add_argument("csv", help="Path to <algo>_summary.csv")
    p.add_argument("--s", type=float, required=True, help="Value of s to select")
    p.add_argument("--beta", default="best",
                   help="'best' (beta with smallest median for each gamma and l) or a numeric value")
    p.add_argument("--score", default="last", choices=["last", "best"],
                   help="Which score family to plot (last_* or best_* columns)")
    p.add_argument("--d", type=int, default=None, help="Filter on dimension")
    p.add_argument("--budget", type=int, default=None, help="Filter on budget")
    p.add_argument("--last-frac", type=float, default=None, help="Filter on last_frac")
    p.add_argument("--clip", type=float, default=1.0, help="Clip values above this (and NaN) to this value")
    p.add_argument("--ylog", action="store_true", help="Log scale on the y axis")
    p.add_argument("--out", default="./", help="Output image path (shown on screen if omitted)")
    return p.parse_args()


def load(args):
    df = pd.read_csv(args.csv)
    for col in ["gamma", "beta", "l", "s", "d", "budget", "last_frac"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if args.d is not None:
        df = df[df["d"] == args.d]
    if args.budget is not None:
        df = df[df["budget"] == args.budget]
    if args.last_frac is not None:
        df = df[np.isclose(df["last_frac"], args.last_frac)]
    df = df[np.isclose(df["s"], args.s)]

    # keep only the most recent run for identical (gamma, beta, l, s)
    df = df.sort_values("timestamp").drop_duplicates(["gamma", "beta", "l", "s"], keep="last")

    # clip median/percentiles; NaN (diverged runs) -> clip value
    cols = [f"{args.score}_{k}" for k in ("median", "p05", "p95")]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(args.clip).clip(upper=args.clip)
    return df, cols


def select_beta(df, beta, med_col):
    if beta == "best":
        # for each (l, gamma) take the row with the smallest (clipped) median over beta
        idx = df.groupby(["l", "gamma"])[med_col].idxmin()
        return df.loc[idx]
    b = float(beta)
    return df[np.isclose(df["beta"], b)]


def main():
    args = get_args()
    df, (med, lo, hi) = load(args)
    if df.empty:
        raise SystemExit("No rows match the given filters.")
    df = select_beta(df, args.beta, med)
    if df.empty:
        raise SystemExit(f"No rows for beta={args.beta}. Available: {sorted(set(pd.read_csv(args.csv)['beta']))}")

    ls = sorted(df["l"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(ls)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for l, c in zip(ls, colors):
        g = df[df["l"] == l].sort_values("gamma")
        ax.plot(g["gamma"], g[med], marker="o", ms=4, color=c, label=f"l={l:g}", rasterized=True)
        ax.fill_between(g["gamma"], g[lo], g[hi], color=c, alpha=0.15, rasterized=True)

    ax.set_xscale("log")
    if args.ylog:
        ax.set_yscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel(f"normalized value ({args.score}), median with p05-p95 band")
    beta_txt = "best beta per point" if args.beta == "best" else f"beta={args.beta}"
    ax.set_title(f"s={args.s:g}, {beta_txt}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, title="l")
    fig.tight_layout()

#    if args.out:
    fig.savefig(f"{args.out}/change_gamma.pdf", dpi=200, bbox_inches="tight")
    print(f"Saved in {args.out}")
#    else:
#        plt.show()


if __name__ == "__main__":
    main()