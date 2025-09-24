import os
import sys 
import numpy as np 

import matplotlib.pyplot as plt

import matplotlib

sys.path.append("..")

from comparisons_utils import get_params_grid

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



results_dir = sys.argv[1]
output_dir = sys.argv[2]
target_fun = sys.argv[3]
d = int(sys.argv[4])    
num_reps = int(sys.argv[5])
reg = float(sys.argv[6])
#n = int(sys.argv[6])

method_to_label = {
#    'fd_gaus' : 'FD [gaussian]', 
#    'fd_sph' : 'FD [spherical]', 
#    'fd_orth' : 'OZD', 
    'de_2p' : 'DE',
    'cmaes' : 'CMA-ES',
    'cbo' : 'CBO',
    'pso' : 'PSO',
    'szlan' : 'SZ-LAN'
}

methods = ['pso', 'de_2p', 'cbo', 'szlan']# 'cmaes', [ 'de_2p', 'pso', 'szlan'] #['fd_gaus', 'fd_sph', 'fd_orth', 'szlan'] #['de_2p', 'pso', 'cmaes', 'szlan']

os.makedirs(output_dir + f"/{target_fun}_{reg}/{d}", exist_ok=True)

# reps = range(num_reps)
# if d > 5:
#     num_directions = [1, 2, d//2, d]
# else:
#     num_directions = [1, d//2, d]
# gammas =  np.linspace(0.001, 0.01, 10) #[1e-5, 1e-3, 1e-2, 1e-1, 0.25, 1.0, 10.0] 
# betas = [1.0, 5.0, 10.0, 50, 100, 200] 
# num_particles = [2, 5, 10, 100]
# f"{optimizer_name}_{target_name}_{d}_" + "_".join([str(x) for x in params[:-1]]) + ".txt"

def read_results(fname):

    normalized_optimality_gap = []
#    print("READING ", f"{results_dir}/szlan_results/comparison/{fname}_{reg}")
    with open(f"{results_dir}/szlan_results/comparison/{fname}", "r") as f:
        #                f.write(f"{rep},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f}\n")
        for line in f.readlines():
            splitted = line.split(",")
            opt_gap = (float(splitted[-5].replace('[', '').replace(']', '')) - float(splitted[-1].replace('[', '').replace(']', ''))) / (float(splitted[-3].replace('[', '').replace(']', '')) - float(splitted[-1].replace('[', '').replace(']', ''))) 
            if np.isnan(opt_gap) or opt_gap > 1.0:
                opt_gap = 1.0
            normalized_optimality_gap.append(opt_gap)
    return normalized_optimality_gap

def read_trace(fun_name, method, fname, num_reps, alg_cost=1):
    trace = []
    for r in range(num_reps):
        trace.append([])
        current_best = None
        with open(f"{results_dir}/szlan_results/comparison/{fun_name}_{reg}/{method}/traces/{fname}_{r}_trace.txt", "r") as f:
            for line in f.readlines():
                splitted = line.split(",")
                current_value = float(splitted[0].replace('[', '').replace(']', ''))
                init_value = float(splitted[1].replace('[', '').replace(']', ''))
                min_value = float(splitted[4].replace('[', '').replace(']', ''))
                opt_value = (current_value - min_value) / (init_value - min_value)
                if current_best is None or opt_value < current_best:
                    current_best = opt_value
                trace[r] += [current_best for _ in range(int(splitted[2].replace('[', '').replace(']', '')))]
    trace = np.array(trace).reshape(num_reps, -1)
    return np.median(trace, axis=0), np.percentile(trace, 95, axis=0), np.percentile(trace, 5, axis=0)
#    return np.mean(trace, axis=0), np.std(trace, axis=0)

fig, ax = plt.subplots(1, 1)
ax.set_title(f"{target_fun} [$d = {d}$]", fontsize=20)

for i, method in enumerate(methods):
    param_grid = get_params_grid(method, d, num_reps)
    best_param, best_param_value = None, None
    label=method
#    if method in ['fd_gaus', 'fd_sph', 'fd_orth']:
#        method = 'szlan'
    for param_config in param_grid:
        fname = f"{target_fun}_{reg}/{method}/{method}_{target_fun}_{d}_" + "_".join([str(x) for x in param_config[:-1]]) + ".txt"
        if not os.path.exists(f"{results_dir}/szlan_results/comparison/{fname}"):
            continue
        opt_gap = np.mean(read_results(fname))
        if best_param_value is None or opt_gap < best_param_value:
            best_param_value = opt_gap
            best_param = param_config

#de_2p_StyblinkskiTang_50_1.0_1.0_1.0
    fname = f"{method}_{target_fun}_{d}_" + "_".join([str(x) for x in best_param[:-1]])
    print(method, best_param, best_param_value)

    alg_cost = ((best_param[0]) ) * best_param[-2] if method in ['fd_gaus', 'fd_sph', 'fd_orth',"szlan"] else 1
    trace_mu, trace_highp, trace_lowp = read_trace(target_fun, method, fname, num_reps, alg_cost)
#    trace_mu, trace_std = read_trace(target_fun, method, fname, num_reps, alg_cost)
    ax.plot(range(len(trace_mu)), trace_mu, '-', label=f"{method_to_label[label]}", lw=3, rasterized=True)
    ax.fill_between(range(len(trace_mu)), trace_lowp, trace_highp, alpha=0.2, rasterized=True)
#    ax.fill_between(range(len(trace_mu)), trace_mu - trace_std, trace_mu + trace_std, alpha=0.2, rasterized=True)

ax.set_yscale('log')
#ax.set_xscale('log')
ax.legend(loc='lower left')
ax.set_xlabel("# Function evaluations", fontsize=16)
ax.set_ylabel("$\\frac{\\min_{i} F(x_k^i) - F^*}{\\min_{i} F(x_0^i) - F^*}$", fontsize=16)
fig.savefig(f"{output_dir}/{target_fun}_{reg}/{d}/{target_fun}_{method}_{d}_trace.pdf", bbox_inches='tight')
plt.close(fig)
