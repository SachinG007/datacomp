#!/bin/bash

folder_path=$1

# Sort files based on step number and iterate through them
pids=()
count=0
# ckpt_folder_path="$folder_path/checkpoints/"
# for file in $(ls $folder_path/checkpoints | grep '\.pt$' | sort -n -k2 -t_ -); do
for file in $(ls $folder_path/checkpoints | grep -E 'epoch_1.pt$'); do
    
    ckpt_path="$folder_path/checkpoints/$file"
    echo "folder path is below"
    echo $folder_path
    
    echo "ckpt path is below"
    echo $ckpt_path
    
    # Run the evaluation script
    CUDA_VISIBLE_DEVICES=$count python evaluate.py --train_output_dir $folder_path --use_model "ViT-B-32 $ckpt_path" &
    
    pids+=($!)
    count=$((count+1))
    #count modulo 8
    
    if [ $count -eq 3 ]; then
        wait "${pids[@]}"
        
        # Reset counter and PIDs
        count=0
        pids=()
    fi
    
    
done
wait "${pids[@]}"
