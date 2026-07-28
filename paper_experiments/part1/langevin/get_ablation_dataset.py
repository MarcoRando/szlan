import numpy as np 

from itertools import product


dimensions = [10, 50, 100, 1000]
gammas = np.logspace(-7, 0, 20)
betas  = np.logspace(-2, 3, 20)

grid = list(product(gammas, betas))
targets = [ 'rastrigin', 'levy', 'ackley', 'rosenbrock', 'griewank', 'least_squares']
with open(f"./langevin_ablation_dataset.csv", 'w') as f_dataset:    
    for target_name in targets:
        for d in dimensions:            
            for line in grid:
                f_dataset.write(f"{target_name},{d},{line[0]},{line[1]}\n")
                f_dataset.flush()
        













