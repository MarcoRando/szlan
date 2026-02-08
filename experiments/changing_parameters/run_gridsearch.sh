#!/bin/bash

budget=1000000
seed=8412

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output path> <num workers> <repetitions> <device>"
    exit 1
fi

out_dir=$1
num_workers=$2
reps=$3
device=$4



fun_names=('ZigZag' 'ZigZagSmooth')
dimensions=(100) 
#  50 25)
# 50)

for fun_name in "${fun_names[@]}"; do
    for d in "${dimensions[@]}"; do
        echo "[>>] Running grid search for $fun_name [d = $d]..."
        python3 grid_search.py $fun_name --d $d --budget $budget --reps $reps --h 1e-5 --num-workers $num_workers --out-dir $out_dir --device $device --seed $seed
    done
done