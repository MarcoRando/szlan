import sys
from itertools import product

import numpy as np

dir_type = sys.argv[1]

dimensions = [10, 50, 100]#, 1000]
gammas = np.logspace(-7, 0, 20)
betas  = np.logspace(-2, 3, 20)

targets = [ 'rastrigin', 'levy', 'ackley']#, 'rosenbrock', 'griewank', 'least_squares']
with open(f"./szlan_ablation_dataset_{dir_type}_long.csv", 'w') as f_dataset:    
    for target_name in targets:
        for d in dimensions:            
            s_vals =  [1, d//5, d//3, d//2, int((2/3) *d), int((4/5) *d), d]
            grid = list(product(gammas, betas, s_vals))
            for line in grid:
                f_dataset.write(f"{target_name},{d},{dir_type},{line[0]},{line[1]},{line[2]}\n")
                f_dataset.flush()
        













