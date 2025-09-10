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
n = int(sys.argv[5])

os.makedirs(output_dir + f"/{target_fun}/{d}", exist_ok=True)

reps = range(5)
num_reps = len(reps)
if d >= 10:
    num_directions = [2, d//3, d//2, int(4/5 * d), d]
elif d > 5:
    num_directions = [2, d//2, d]
else:
    num_directions = [d//2, d]
gammas =  [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0] #np.logspace(-7, 0, 5)
betas =[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5] # np.logspace(-7, 0, 8) #np.logspace(-5, 0, 5)


def read_results(fun_name, d, l, gamma, beta):

    traces = []
    for rep in reps:
        trace = []
        f0 = None
        with open(f"{results_dir}/szlan_results_old/changing_parameters/{fun_name}/traces/{fun_name}_{d}_{l}_{gamma}_{beta}_{n}_{rep}_trace.txt", "r") as f:
            for (i, line) in enumerate(f.readlines()):
                splitted = line.split(",")
                if f0 is None:
                    f0 = float(splitted[0])
                opt_ratio = float(splitted[0])# (float(splitted[0]) - float(splitted[-1]))/ (f0 - float(splitted[-1]))
                if np.isnan(opt_ratio) or opt_ratio > 1:
                    opt_ratio = 1.0
                trace.append(opt_ratio)
        traces.append(trace)    
        #print(trace)
    traces = np.array(traces).reshape(num_reps, -1)#[:,:40000]
    return np.mean(traces, axis=0), np.std(traces, axis=0)



fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title(f"{target_fun} [d = {d}]")
for l in num_directions:
    current_best = None
    current_best_mu_vals, current_best_std_vals = [], []
    for gamma in gammas:
        for beta in betas:
            mu, std = read_results(target_fun, d, l, gamma, beta)
            if current_best is None or np.min(mu) < current_best:
                current_best = np.min(mu)
                current_best_mu_vals = mu
                current_best_std_vals = std
    current_best_mu_vals = np.array(current_best_mu_vals)
    current_best_std_vals = np.array(current_best_std_vals)
    ax.plot(range(len(current_best_mu_vals)), current_best_mu_vals, label=f"$\\ell=${l}", rasterized=True)
    ax.fill_between(range(len(current_best_mu_vals)), current_best_mu_vals - current_best_std_vals, current_best_mu_vals + current_best_std_vals, alpha=0.2, rasterized=True)
    print("[->] Plotted for l =", l)
ax.legend()
ax.set_yscale("log")
fig.savefig(f"{output_dir}/{target_fun}_{d}_{n}_trace.png", bbox_inches='tight')
plt.close()