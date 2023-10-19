#!/bin/bash
python evaluate.py --train_output_dir logs/logs/mediumscale_nofilter_5x/ --use_model "ViT-B-32 /home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x/checkpoints/stepwise/model_step_6250_epoch_0.pt"
python evaluate.py --train_output_dir logs/logs/mediumscale_nofilter_5x/ --use_model "ViT-B-32 /home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x/checkpoints/stepwise/model_step_12500_epoch_0.pt.pt"
python evaluate.py --train_output_dir logs/logs/mediumscale_nofilter_5x/ --use_model "ViT-B-32 /home/sachingo/datacomp/logs/logs/mediumscale_nofilter_5x/checkpoints/stepwise/model_step_12500_epoch_0.pt.pt"
