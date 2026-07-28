import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dimensions = [10, 50, 100]#, 1000]
# gammas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] #np.logspace(-7, 0, 20)
# betas  = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0] #np.logspace(-2, 3, 15)

gammas = np.logspace(-7, 0, 20)
betas  = np.logspace(-2, 3, 20)



functions = ['ackley', 'rastrigin', 'levy']

result_dir = sys.argv[1]

lb = sys.argv[2] 


assert lb in ['gamma', 'beta']

# Impact of \gamma and \beta

l_to_labels = ['1', '\\frac{1}{5}d', '\\frac{1}{3}d', '\\frac{1}{2}d', '\\frac{2}{3}d', '\\frac{4}{5}d', 'd']
#l_to_labels = ['1',  '\\frac{1}{3}d', '\\frac{1}{2}d', '\\frac{2}{3}d',  'd']


fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i in range(3): # gamma or beta    
    d = dimensions[i]
    num_dirs = [1, d//5, d//3, d//2, int((2/3) *d), int((4/5) *d), d]
#    num_dirs = [1, d//3, d//2, int((2/3) *d), d]
    for j in range(len(functions)):

        if j == 0:
            axes[i, j].text(
                0.05, 0.85,
                rf"$d = {d}$",
                transform=axes[i,j].transAxes,   # coordinates relative to the axes
                fontsize=10,
                verticalalignment="top",
                bbox={
                    'boxstyle': "round",
                    'facecolor': "white",
                    'edgecolor': "black",
                    'alpha': 0.8
                }
            )
        if not os.path.exists(f"{result_dir}/{functions[j]}_{d}_spherical_tested.log"):
            continue
        print(f"[--] processing: {result_dir}/{functions[j]}_{d}_spherical_tested.log")
        data = pd.read_csv(f"{result_dir}/{functions[j]}_{d}_spherical_tested.log", header=None)
        for (l_id, l) in enumerate(num_dirs):        
            mu_vals, std_vals = [], []
            
            data_l = data[data[2] == l] #if lb == 'l' else data[data[3] == l]
            x_used = []
            iter_list = gammas if lb == 'gamma' else betas
            for gamma in iter_list:
                data_lg = data_l[data_l[0] == gamma] if lb =='gamma' else data_l[data_l[1] == gamma] 
                data_lg = data_lg[[6, 7]].values
                if data_lg.shape[0] == 0:
                    continue
                x_used.append(gamma)
                idx_best = np.argmin(data_lg[:, 0] + data_lg[:, 1])
            
                if data_lg[idx_best, 0] < 1:
                    mu_elem, std_elem = data_lg[idx_best, 0], data_lg[idx_best, 1]  
                else:
                    mu_elem, std_elem = 1.0, 0.0
                mu_vals.append(mu_elem) ; std_vals.append(std_elem)

            mu_vals, std_vals = np.array(mu_vals), np.array(std_vals)
            
            lcb, ucb = mu_vals - std_vals, mu_vals + std_vals 

            lcb[lcb < 0] = mu_vals[lcb < 0] - 1e-7
            print("NUM X USED: ", len(x_used))
            axes[i, j].plot(x_used, mu_vals, 'o-', lw=3, label=f"$s = {l_to_labels[l_id]}$", rasterized=True)
            axes[i, j].fill_between(x_used, lcb, ucb, lw=3, alpha=0.4, rasterized=True)
        if i == 0:
            axes[i,j].set_title(functions[j].capitalize(), fontsize=18)
        if j ==0:
            axes[i,j].set_ylabel('$\\frac{F(x_k) - \min F}{F(x_0) - \min F}$', fontsize=16)
        if lb == 'gamma':
            axes[i,j].set_xlabel('$\gamma$', fontsize=14)
        else:
            axes[i,j].set_xlabel('$\\beta$', fontsize=14)
        axes[i,j].set_xscale('log')
        axes[i,j].set_yscale('log')

axes[0,0].legend(loc='upper center', fontsize=12, ncol=len(num_dirs), frameon=True, edgecolor="black", bbox_to_anchor=(1.75, -2.850))
fig.subplots_adjust(wspace=0.35,hspace=0.3)

if lb == 'gamma':
    fig.savefig("./change_l_gamma.pdf", bbox_inches='tight')
else:
    fig.savefig("./change_l_beta.pdf", bbox_inches='tight')
plt.close(fig)








from matplotlib.colors import Normalize

norm = Normalize(vmin=0, vmax=1)

fig, axes = plt.subplots(3, 3, figsize=(14, 12), subplot_kw={"projection": "3d"})

for i in range(3): # gamma or beta    
    d = dimensions[i]
#    num_dirs = [1, d//5, d//3, d//2, int((2/3) *d), int((4/5) *d), d]
    num_dirs = [1, d//3, d//2, int((2/3) *d), d]

    for j in range(len(functions)):

        if j == 0:
            axes[i, j].text(
                -0.002, 0.75, 0.0,
                s=rf"$d = {d}$",
                transform=axes[i,j].transAxes,   # coordinates relative to the axes
                fontsize=10,
                verticalalignment="top",
                bbox={
                    'boxstyle': "round",
                    'facecolor': "white",
                    'edgecolor': "black",
                    'alpha': 0.8
                }
            )
            
        if not os.path.exists(f"{result_dir}/{functions[j]}_{d}_spherical_tested.log"):
            continue


        data = pd.read_csv(f"{result_dir}/{functions[j]}_{d}_spherical_tested.log", header=None)
        hmap = np.zeros((len(gammas), len(betas)))

            #        f.write(f"{gamma},{beta},{s},{mu_values[idx_best]},{std_values[idx_best]},{idx_best},{lst_500_mu[best_last_500]},{lst_500_std[best_last_500]},{num_iters - 100 + best_last_500}\n")
        for (gamma_id, gamma) in enumerate(gammas):
            data_gamma = data[data[0] == gamma] 
            for (beta_id, beta) in enumerate(betas):
                data_gb = data_gamma[data_gamma[1] == beta]
                data_l = data_gb[[6, 7]].values
                if len(data_l) == 0:
                    continue
#                print(len(data_l))
                idx_best = np.argmin(data_l[:, 0] + data_l[:, 1])
                if data_l[idx_best, 0] < 1:
                    mu_elem, std_elem = data_l[idx_best, 0], data_l[idx_best, 1]  
                else:
                    mu_elem, std_elem = 1.0, 0.0
                hmap[gamma_id, beta_id] = mu_elem

        G, B = np.meshgrid(gammas, betas)
        
        print(f"min: {np.min(np.log10(hmap))}")
        im = axes[i, j].plot_surface(np.log10(G), np.log10(B),  np.log10(hmap).T, cmap="magma", vmin=-1.8, vmax=0.0)
        axes[i,j].invert_yaxis()
#        im = axes[i, j].imshow(hmap.T, origin='lower',  norm=norm)                
        
#        axes[i, j].set_xticks(range(len(gammas)), labels=gammas, rotation=45)
#        axes[i, j].set_yticks(range(len(betas)) , labels=betas)
        
        if i == 0:
            axes[i,j].set_title(functions[j].capitalize(), fontsize=20)
#        if j ==0:

        axes[i,j].set_xlabel('$\gamma$', fontsize=16)
        axes[i,j].set_ylabel('$\\beta$', fontsize=16)
#        axes[i,j].set_zlabel('$\\log_{10} \\frac{F(x_k) - \min F}{F(x_0) - \min F}$', fontsize=10)
        axes[i,j].set_zlim([-2.0, 0.0])
#        axes[i,j].set_xscale('log')
#        axes[i,j].set_yscale('log')

#fig.subplots_adjust(wspace=0.5, hspace=1.0)
fig.tight_layout()
cbar = fig.colorbar(
    im,
    ax=axes,            # all axes
    location="bottom",   # or "bottom"
    shrink=0.9,
    pad=0.08
)
cbar.set_label("$\\log_{10} \\frac{F(x_k) - \min F}{F(x_0) - \min F}$", fontsize=18)

fig.savefig("./change_gamma_beta.pdf", bbox_inches='tight')
plt.close(fig)
