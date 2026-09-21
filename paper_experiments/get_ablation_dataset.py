import sys 
import numpy as np 

from itertools import product

d = int(sys.argv[1])

dir_type = sys.argv[2]

function_names = ['rastrigin', 'ackley', 'levy']
gammas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1]
betas = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
num_directions = [1, int((1/5) * d), int((1/3) * d), d//2, int((2/3) * d), int((4/5) * d), d]

with open(f"./zlan_dataset_{d}.csv", 'w') as f:
    
    for (fname, gamma, beta, s) in product(function_names, gammas, betas, num_directions):
        f.write(f"{fname},{dir_type},{d},{gamma},{beta},{s}\n")
        f.flush()