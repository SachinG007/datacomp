#!/bin/bash

folder_path=$1

# Sort files based on step number and iterate through them
pids=()
count=0
for file in $(ls $folder_path | sort -n -k2 -t_ -); do
    ckpt_path="$folder_path/checkpoints/$file"
    
    # Run the evaluation script
    python "evaluate.py --train_output_dir $folder_path --use_model \"ViT-B-32 $ckpt_path\""
    
    pids+=($!)
    count=$((count+1))
    if [ $count -eq 8 ]; then
        wait "${pids[@]}"
        
        # Reset counter and PIDs
        count=0
        pids=()
    fi
    
    
done
wait "${pids[@]}"