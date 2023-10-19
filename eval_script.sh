#!/bin/bash

folder_path=$1

# Sort files based on step number and iterate through them
for file in $(ls $folder_path | sort -n -k2 -t_ -); do
    ckpt_path="$folder_path/checkpoints/$file"
    
    # Run the evaluation script
    python evaluate.py --train_output_dir $folder_path --use_model \"ViT-B-32 $ckpt_path\"
done
