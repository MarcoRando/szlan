#!/bin/bash

budget=100000000
seed=123456

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output path> <num workers> <repetitions>"
    exit 1
fi


num_workers=$2
reps=$3

fun_names=('Ackley' 'Levy' 'StyblinkskiTang' 'Griewank' 'Rosenbrock')
dimensions=(5 10 25 50)

for fun_name in "${fun_names[@]}"; do
    for d in "${dimensions[@]}"; do
        echo "[>>] Running grid search for $fun_name [d = $d]..."
        nohup python3 grid_search.py $fun_name --d $d  --budget $budget --reps $reps --num-workers $num_workers --out-dir $1 --seed $seed 
        wait
    done
done