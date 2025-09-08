import os
import sys 
import numpy as np 

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


results_dir = sys.argv[1]
output_dir = sys.argv[2]
target_fun = sys.argv[3]
d = int(sys.argv[4])    
num_reps = int(sys.argv[5])


os.makedirs(output_dir + f"/{target_fun}", exist_ok=True)

reps = range(num_reps)
gammas =  [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
betas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
num_directions = [2, d//3, d//2, int(4/5 * d), d]

def read_results(fun_name, d, l, gamma, beta):

    normalized_optimality_gap = []
    with open(f"{results_dir}/szlan_results/changing_parameters/{fun_name}/{fun_name}_{d}_{l}_{gamma}_{beta}.txt", "r") as f:
        #                f.write(f"{rep},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f}\n")
        for line in f.readlines():
            splitted = line.split(",")
            opt_gap = (float(splitted[-3]) - float(splitted[-1])) / (float(splitted[-2]) - float(splitted[-1]))
            if np.isnan(opt_gap) or opt_gap > 1.0:
                opt_gap = 1.0
            normalized_optimality_gap.append(opt_gap)
    return normalized_optimality_gap


results_map = np.ones((len(num_directions), len(gammas), len(betas)))

for i, l in enumerate(num_directions):
    for j, gamma in enumerate(gammas):
        for k, beta in enumerate(betas):
            results_map[i, j, k] = np.mean(read_results(target_fun, d, l, gamma, beta))

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{target_fun} [$d = {d}, \\ell={l}$]")
    im = ax.imshow(results_map[i].T, cmap='viridis', interpolation='bilinear', origin='lower') #, vmin=0.0, vmax=1.0)
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
    fig.savefig(f"{output_dir}/{target_fun}/results_map_{d}_{l}.pdf", bbox_inches='tight')
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

    ax.set_xlabel("gamma")
    ax.set_ylabel("normalized optimality gap")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(f"{output_dir}/{target_fun}/normalized_optimality_gap_{d}_{l}_changing_beta.pdf", bbox_inches='tight')
    plt.close(fig)


for beta in betas:
    results = []
    for l in num_directions:
        results.append([read_results(target_fun, d, l, gamma, beta) for gamma in gammas])


    results = np.array(results).reshape(len(num_directions), len(gammas), num_reps)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(results)):
        mu_ris, sigma_ris = np.mean(results[i, :, :], axis=-1), np.std(results[i, :, :], axis=-1)
        ax.plot(gammas, mu_ris, '-o', label=f"$\\ell=$ {num_directions[i]}", rasterized=True)
        ax.fill_between(gammas, mu_ris - sigma_ris, mu_ris + sigma_ris, alpha=0.2, rasterized=True)

    ax.set_xlabel("gamma")
    ax.set_ylabel("normalized optimality gap")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(f"{output_dir}/{target_fun}/normalized_optimality_gap_{d}_{beta}_changing_l.pdf", bbox_inches='tight')
    plt.close(fig)