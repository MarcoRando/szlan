#!/bin/bash


if [ $# -ne 3 ]; then
    echo "Usage: $0 input_file out_directory num_parallel_processes"
    exit 1
fi

input_file="$1"
out_dir="$2"
max_parallel=$3

budget=1000000
reps=5
device='cuda'

while IFS= read -r line || [ -n "$line" ]; do

    IFS=',' read -ra fields <<< "$line"

    while [ $(ps aux | grep ablation.py | wc -l) -ge $max_parallel ]; do
        sleep 5 
    done

    fname=${fields[0]}
    dir_type=${fields[1]}
    d=${fields[2]}
    gamma=${fields[3]}
    beta=${fields[4]}
    s=${fields[5]}

    echo "Running: $fname - $d - $gamma - $beta - $s"

    python3 ablation.py $fname --d $d --gamma $gamma --beta $beta --s $s --dir-type $dir_type --budget $budget --reps $reps --out-dir $out_dir --device $device &

done < "$input_file"