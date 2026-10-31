import torch

from synthetic_targets import Alpine1, Ackley, Rastrigin

import matplotlib.pyplot as plt


d = 1
n = 1000


device='cpu'
target_1 = Alpine1(d=d, device=device)
target_2 = Ackley(d=d, device=device)
target_3 = Rastrigin(d = d, device = device)


X = torch.linspace(-10, 10, n).view(n, d)
Y_1, Y_2, Y_3 = target_1(X), target_2(X), target_3(X)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(X, Y_1, '-', c='black', rasterized=True)
ax2.plot(X, Y_2, '-', c='black', rasterized=True)
ax3.plot(X, Y_3, '-', c='black', rasterized=True)

minimizers = [(target.x_star.item(), target(target.x_star).item()) for target in [target_1, target_2, target_3]]
titles = [target_1.name, target_2.name, target_3.name]


for (ax, title, minim) in zip([ax1, ax2, ax3], titles, minimizers):
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("$x$", fontsize=16)
    ax.plot(minim[0], minim[1], 'o', c='darkred', label="$F(x^*)$")

ax1.set_ylabel("$F(x)$", fontsize=16)
ax1.legend(fontsize=12)

fig.tight_layout()
fig.savefig(f"synthetic_targets.pdf", bbox_inches='tight')
plt.close(fig)