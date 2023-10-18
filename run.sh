#!/bin/bash
python evaluate.py --train_output_dir logs/logs/mediumscale_nofilter_5x_sac/ --use_model "ViT-B-32 /home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x_sac/checkpoints/epoch_3.pt"
