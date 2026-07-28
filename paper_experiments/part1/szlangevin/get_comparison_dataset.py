import sys
from itertools import product

dir_type = sys.argv[1]

d = 100 #[10, 50, 100]#, 1000]
s_vals =  [d, int((2/3) * d), d//2, d//3, 1]# [1, d//5, d//3, d//2, int((2/3) *d), int((4/5) *d), d]
gammas = [0.0001, 0.001, 0.01, 0.1] #np.logspace(-7, 0, 20)
betas  = [0.1, 1.0, 10.0, 50.0, 100.0, 1000.0] #np.logspace(-2, 3, 20)

methods = ['zlan', 'fd']

targets = [ 'rastrigin', 'levy', 'ackley']#, 'rosenbrock', 'griewank', 'least_squares']
with open(f"./szlan_comparison_dataset_{dir_type}_long.csv", 'w') as f_dataset:    
    for target_name in targets:
        for method in methods:           
            if method == 'fd':
                grid = list(product(gammas, [1000.0], s_vals))
            else:
                grid = list(product(gammas, betas, s_vals))
            for line in grid:
                f_dataset.write(f"{target_name},{d},{method},{dir_type},{line[0]},{line[1]},{line[2]}\n")
                f_dataset.flush()
        













