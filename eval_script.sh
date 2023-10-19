#!/bin/bash

folder_path="/home/sachingo/datacomp/logs/clipbucket_30p_to_40p/checkpoints"

# Sort files based on step number and iterate through them
for file in $(ls $folder_path | sort -n -k2 -t_ -); do
	ckpt_path="$folder_path/$file"
	    
	# Run the evaluation script
	echo "python evaluate.py --train_output_dir logs/clipbucket_30p_to_40p/ --use_model \"ViT-B-32 $ckpt_path\""
done
