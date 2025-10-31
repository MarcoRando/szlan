import sys
import numpy as np 


import matplotlib.pyplot as plt


results_dir = f"{sys.argv[1]}/szlan_results/comp_langevin"

fun_name = sys.argv[2]

d = int(sys.argv[3])

num_directions = [2, d//3, d//2, int(2/3 * d), d]
num_particles = int(sys.argv[4])
h = 1e-5

fig, ax = plt.subplots(1, 1)

ax.set_title(f"{fun_name} [$d$ = {d}, $n$ = {num_particles}]")

for i in range(len(num_directions)):
    data = np.loadtxt(f"{results_dir}/{fun_name}/szlan_{fun_name}_{d}_{num_directions[i]}_{num_particles}_{h}_constant_optgaps.log", delimiter=',')
    costs = data[:, -2]
    gaps = data[:, :-2]
    mu_gap, std_gap = np.mean(gaps, axis=1), np.std(gaps, axis=1)


    expanded_mu, expanded_std = [], []#np.array([x for x in mu_gap ])
    for j in range(len(mu_gap)):
#        if j ==0 or  mu_gap[j] < expanded_mu[-1]:
        expanded_mu += [mu_gap[j]] * int(costs[j])
        expanded_std += [std_gap[j]] * int(costs[j]) #num_particles * (num_directions[i] + 1) #(int(costs[j]))

    expanded_mu = np.array(expanded_mu)[:50000]
    expanded_std = np.array(expanded_std)[:50000]
    ax.plot(range(len(expanded_mu)), expanded_mu, '-', label=f"$\\ell = ${num_directions[i]}", rasterized=True)
    ax.fill_between(range(len(expanded_mu)), expanded_mu - expanded_std, expanded_mu + expanded_std, alpha=0.4, rasterized=True)

ax.set_yscale('log')
#ax.set_xscale('log')
ax.legend()
ax.set_ylabel("$\\frac{\\min_{i,j} F(x_i^j) - F^*}{\\min_j F(x_0^j) - F^*}$")
ax.set_xlabel("evaluations")
fig.savefig(f"{fun_name}_{d}_{num_particles}_comparison_with_lan.pdf", bbox_inches='tight')
plt.close(fig)