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

fun_names=('Ackley' 'Levy' 'StyblinkskiTang' 'Griewank' 'Rastrigin' 'Rosenbrock')
regularization=(0) #1)
# 10)
#('Ackley')
# 'Levy' 'StyblinkskiTang' 'Griewank' 'Rosenbrock')
dimensions=(10 50 100)
num_particles=(2 5 10 100)
#  200)

#(100 200) 
#(10 25 50) 
#(5 10 25)
MAX_JOBS=3

for num_part in "${num_particles[@]}"; do
    for d in "${dimensions[@]}"; do
        for fun_name in "${fun_names[@]}"; do
            for reg in "${regularization[@]}"; do
    #            echo "[$(ps aux  | grep grid_search | wc -l) -ge $MAX_PROCS )"
                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    echo "Waiting for grid search to finish..."
                    echo "[current jobs: $(jobs -r | wc -l)]"
                    echo "[current processes: $(ps aux  | grep grid | wc -l)]"
                    echo "[max jobs: $MAX_JOBS]"
                    sleep 1
                done
                python3 grid_search.py 'szlan' $fun_name --d $d --reg $reg --budget $budget --num_particles $num_part --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
    #            python3 grid_search.py 'cmaes' $fun_name --d $d --reg $reg --budget $budget --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                python3 grid_search.py 'cbo' $fun_name --d $d --reg $reg --budget $budget --num_particles $num_part --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                python3 grid_search.py 'de_2p' $fun_name --d $d --reg $reg --budget $budget --num_particles $num_part --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                python3 grid_search.py 'pso' $fun_name   --d $d --reg $reg --budget $budget --num_particles $num_part --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed & 
                # while (( $(ps aux  | grep grid | wc -l) >= $MAX_PROCS )); do
                #     echo "Waiting for grid search to finish..."
                #     echo "[current processes: $(ps aux  | grep grid | wc -l)]"
                #     echo "[max processes: $MAX_PROCS]"
                #     sleep 1
                # done

                # echo "[>>] Running grid search for $fun_name [d = $d]..."
                # nohup python3 grid_search.py 'szlan' $fun_name --d $d --reg $reg --budget $budget --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                # nohup python3 grid_search.py 'cmaes' $fun_name --d $d --reg $reg --budget $budget --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                # nohup python3 grid_search.py 'de_2p' $fun_name --d $d --reg $reg --budget $budget --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
                # nohup python3 grid_search.py 'pso' $fun_name   --d $d --reg $reg --budget $budget --reps $reps --num-workers $num_workers --out-dir $out_dir --seed $seed &
            done
        done
    done 
done