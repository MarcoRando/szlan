#!/bin/bash

#budget=100000000

budget=50000

#100000 

seed=123456

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output path> <num workers> <repetitions>"
    exit 1
fi



out_dir=$1
num_workers=$2
reps=$3
#method=$4

fun_names=('Rosenbrock') #('Ackley' 'Levy' 'StyblinkskiTang' 'Griewank' 'Rastrigin' 'Rosenbrock')
regularization=(0) #1)
# 10)
#('Ackley')
# 'Levy' 'StyblinkskiTang' 'Griewank' 'Rosenbrock')
dimensions=(10 50 100)
num_particles=(2 5 10 50 100)
num_directions=(0 1 2 3 4) # -> 2, d/3, d/2, 2/3 * d, d
#  200)

#(100 200) 
#(10 25 50) 
#(5 10 25)
MAX_JOBS=3

for num_part in "${num_particles[@]}"; do
    for d in "${dimensions[@]}"; do
        for fun_name in "${fun_names[@]}"; do
            for l in "${num_directions[@]}"; do

                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    echo "Waiting for grid search to finish..."
                    echo "[current jobs: $(jobs -r | wc -l)]"
                    echo "[current processes: $(ps aux  | grep grid | wc -l)]"
                    echo "[max jobs: $MAX_JOBS]"
                    sleep 1
                done
                python3 comparison_with_lan.py $fun_name 'szlan' $d $num_part $out_dir $num_workers $reps $budget $l "constant" 1e-5 & 

            done
        done
    done 
done
python3 comparison_with_lan.py 'lan' $fun_name 'lan' $d $num_part $out_dir $num_workers $reps $budget
