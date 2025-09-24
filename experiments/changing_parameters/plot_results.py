import os
import sys 
import numpy as np 

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



results_dir = sys.argv[1]
output_dir = sys.argv[2]
target_fun = sys.argv[3]
d = int(sys.argv[4])    
num_reps = int(sys.argv[5])
n = int(sys.argv[6])





os.makedirs(output_dir + f"/{target_fun}/{d}", exist_ok=True)

reps = range(num_reps)
if d >= 10:
    num_directions = [2, d//3, d//2, int(4/5 * d), d]
    num_directions_labels = ["$2$", "$\\frac{d}{3}$", "$\\frac{d}{2}$", "$\\frac{4d}{5}$", "$d$"]
elif d > 5:
    num_directions = [2, d//2, d]
    num_directions_labels = ["$2$", "$\\frac{d}{2}$", "$d$"]
else:
    num_directions = [d//2, d]
    num_directions_labels = ["$\\frac{d}{2}$", "$d$"]

gammas =  [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0] #, 10.0, 100.0] #np.logspace(-7, 0, 5)
betas =[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1]#, 1e2, 1e3, 1e4, 1e5]

def read_results(fun_name, d, l, gamma, beta):

    normalized_optimality_gap = []
    with open(f"{results_dir}/szlan_results/changing_parameters/{fun_name}/{fun_name}_{d}_{l}_{gamma}_{beta}_{n}.txt", "r") as f:
        #                f.write(f"{rep},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f}\n")
        for line in f.readlines():
            splitted = line.split(",")
            opt_gap = (float(splitted[-3]) - float(splitted[-1])) / (float(splitted[-2]) - float(splitted[-1]))
            if np.isnan(opt_gap) or opt_gap > 1.0:
                opt_gap = 1.0
            normalized_optimality_gap.append(opt_gap)
    return normalized_optimality_gap

def read_trace(fun_name, d, l, gamma, beta, num_reps):
    trace = []
    for r in range(num_reps):
        trace.append([])
        current_best = None
        with open(f"{results_dir}/szlan_results/changing_parameters/{fun_name}/traces/{fun_name}_{d}_{l}_{gamma}_{beta}_{n}_{r}_trace.txt", "r") as f:
            for line in f.readlines():
                splitted = line.split(",")
                current_value = float(splitted[0])
                init_value = float(splitted[1])
                min_value = float(splitted[2])
                opt_value = (current_value - min_value) / (init_value - min_value)
                if current_best is None or opt_value < current_best:
                    current_best = opt_value
                trace[r] += [current_best for _ in range(l + 1)]
    trace = np.array(trace).reshape(num_reps, -1)
    return np.median(trace, axis=0), np.percentile(trace, 90, axis=0), np.percentile(trace, 5, axis=0)

results_map = np.ones((len(num_directions), len(gammas), len(betas)))
best_for_l = []
best_for_l_gamma_mu = [[] for _ in range(len(num_directions))]
best_for_l_gamma_std = [[] for _ in range(len(num_directions))]
traces = []
for i, l in enumerate(num_directions):

    l_values = []
    best_mean_ = None
    idx_gamma_best, idx_beta_best = None, None
    for j, gamma in enumerate(gammas):
        l_gamma_values = []

        best_mean_gamma = None
        for k, beta in enumerate(betas):
            results_l_gamma_beta = read_results(target_fun, d, l, gamma, beta)
            results_map[i, j, k] = np.mean(results_l_gamma_beta)
            if best_mean_ is None or results_map[i, j, k] < best_mean_:
                best_mean_ = results_map[i, j, k]
                l_values = results_l_gamma_beta
                idx_gamma_best, idx_beta_best = j, k
            if best_mean_gamma is None or results_map[i, j, k] < best_mean_gamma:
                best_mean_gamma = results_map[i, j, k]
                l_gamma_values = results_l_gamma_beta

        best_for_l_gamma_mu[i].append(np.mean(l_gamma_values))
        best_for_l_gamma_std[i].append(np.std(l_gamma_values))

    trace_mu, trace_highp, trace_lowp = read_trace(target_fun, d, l, gammas[idx_gamma_best], betas[idx_beta_best], num_reps)
    traces.append(( trace_mu, trace_highp, trace_lowp))

    best_for_l.append(l_values)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{target_fun} [$d = {d}, \\ell={l}$]")
    im = ax.imshow(results_map[i].T, cmap='viridis', interpolation='bilinear', origin='lower', vmin=0.0, vmax=1.0)
#    ax.plot(idx_gamma_best, idx_beta_best, 'o', color='red', markersize=10)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(betas)))
#    ax.set_yticks(np.arange(len(gammas)))
    ax.set_xticklabels(gammas, rotation=45)
    ax.set_yticklabels(betas)
#    ax.set_yscale("log")
#    ax.set_xscale("log")

    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\beta$")
    fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_results_map_{d}_{l}.pdf", bbox_inches='tight')
    plt.close(fig)


fig, ax = plt.subplots(1, 1)
for i in range(len(traces)):
    trace_mu, trace_highp, trace_lowp = traces[i]
    ax.plot(range(len(trace_mu)), trace_mu, '-', label=f"$\\ell=${num_directions[i]}", lw=3, rasterized=True)
    ax.fill_between(range(len(trace_mu)), trace_lowp, trace_highp, alpha=0.2, rasterized=True)
#    ax.fill_between(range(len(trace_mu)), trace_mu -trace_std, trace_mu + trace_std, alpha=0.2, rasterized=True)
#    ax.fill_between(range(len(trace_mu)), trace_mu - trace_std, trace_mu + trace_std, alpha=0.2, rasterized=True)

ax.set_xlabel("Iteration")
ax.set_ylabel("$\\frac{\\min_{i,k} F(x_k^i) - \\min F}{\\min_{i} F(x_0^i) - \\min F}$")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.legend(loc='upper right')
fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_optimality_gap_{d}_{l}_trace.pdf", bbox_inches='tight')
plt.close(fig)


fig, ax =plt.subplots(1, 1)

for i, l in enumerate(num_directions):
    mu, std = np.array(best_for_l_gamma_mu[i]), np.array(best_for_l_gamma_std[i])
    ax.plot(gammas, mu, '-o', label=f"$\\ell=${l}", color=colors[i], rasterized=True)
    ax.fill_between(gammas, mu - std, mu + std, color=colors[i], alpha=0.2, rasterized=True)

ax.legend()
ax.set_xlabel("$\\gamma$")
ax.set_ylabel("$\\frac{\\min_{i,k}  F(x_k^i) - F^*}{\\min_{i} F(x_0^i) - F^*}$")
ax.set_yscale("log")
ax.set_xscale("log")
fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_optimality_gap_{d}_changing_l_best_beta.pdf", bbox_inches='tight')
plt.close(fig)

print("[--] Plotting l comparison...")
fig, ax = plt.subplots(1, 1)
ax.set_title(f"{target_fun} [$d = {d}$]", fontsize=20)
b = ax.boxplot(best_for_l, positions=np.arange(len(num_directions)), patch_artist=True, tick_labels=num_directions, widths=0.5)

for (i, patch) in enumerate(b['boxes']):
    patch.set_facecolor(colors[i])
#    patch.set_edgecolor(colors[patch['label']])
    patch.set_alpha(0.5)
for patch in b['medians']:
    patch.set_color('darkred')

ax.set_xlabel("$\\ell$", fontsize=18)
ax.set_ylabel("$\\frac{\\min_{i,k}  F(x_k^i) - F^*}{\\min_{i} F(x_0^i) - F^*}$", fontsize=18)

fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_{d}_l_comparison.pdf", bbox_inches='tight')
plt.close(fig)


for l in num_directions:
    results = []
    for beta in betas:
        results.append([read_results(target_fun, d, l, gamma, beta) for gamma in gammas])

    results = np.array(results).reshape(len(betas), len(gammas), num_reps)

    fig, ax = plt.subplots(1, 1)
    for i in range(len(results)):
        mu_ris, sigma_ris = np.mean(results[i, :, :], axis=-1), np.std(results[i, :, :], axis=-1)
        ax.plot(gammas, mu_ris, '-o', label=f"beta={betas[i]}", rasterized=True)
        ax.fill_between(gammas, mu_ris - sigma_ris, mu_ris + sigma_ris, alpha=0.2, rasterized=True)

    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\frac{\\min_{i,k} F(x_k^i) - F^*}{\\min_{i} F(x_0^i) -  F^*}$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_normalized_opt_gap_{d}_{l}_changing_beta.pdf", bbox_inches='tight')
    plt.close(fig)



beta_results = [[] for _ in range(len(num_directions))]

for beta in betas:
    results = []
    for (i, l) in enumerate(num_directions):
        results.append([read_results(target_fun, d, l, gamma, beta) for gamma in gammas])
        best_val_gamma = None
        for (j, gamma) in enumerate(gammas):
            opt_gap = read_results(target_fun, d, l, gamma, beta)
            if best_val_gamma is None or opt_gap < best_val_gamma:
                best_val_gamma = opt_gap
        beta_results[i].append((np.median(best_val_gamma), np.percentile(best_val_gamma, 90), np.percentile(best_val_gamma, 5)))


    results = np.array(results).reshape(len(num_directions), len(gammas), num_reps)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(results)):
        mu_ris, sigma_ris = np.mean(results[i, :, :], axis=-1), np.std(results[i, :, :], axis=-1)
        ax.plot(gammas, mu_ris, '-o', label=f"$\ell=$ {num_directions[i]}", rasterized=True)
        ax.fill_between(gammas, mu_ris - sigma_ris, mu_ris + sigma_ris, alpha=0.2, rasterized=True)

    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\frac{\\min_{i,k} F(x_k^i) - \\min F}{\\min_{i} F(x_0^i) - \\min F}$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_normalized_optimality_gap_{d}_{beta}_changing_l.pdf", bbox_inches='tight')
    plt.close(fig)


print("[--] Plotting beta comparison...")
fig, ax = plt.subplots(1, 1)
for i, l in enumerate(num_directions):
    results = beta_results[i]
    mu, phigh, plow = None, None, None
    mu = np.array([r[0] for r in results])
    phigh = np.array([r[1] for r in results])
    plow = np.array([r[2] for r in results])
    ax.plot(betas, mu, '-o', label=f"$\\ell=$ {num_directions[i]}", rasterized=True)
    ax.fill_between(betas, plow, phigh, alpha=0.2, rasterized=True)
ax.set_xlabel("$\\beta$")
ax.set_ylabel("$\\frac{\\min_{i,k} F(x_k^i) - F^*}{\\min_{i} F(x_0^i) - F^*}$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()

fig.savefig(f"{output_dir}/{target_fun}/{d}/{target_fun}_opt_gap_{d}_changing_beta.pdf", bbox_inches='tight')
plt.close(fig)