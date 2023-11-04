#!/bin/bash
# declare -A path_epochs_dict

# paths_list=("/drive2/datacomp_L14_0.5_to_0.6::/drive2/datacomp_L14_0.4_to_0.5::/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.4_to_0.5::/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1")
# epochs_list=(18 15 8 9)
# length=${#paths_list[@]}

# exp_name=clipbucket_curriculum_adt_128M_5x
# mkdir -p /home/pratyus2/logs/$exp_name/intermediate
# # Loop through the lists and print the elements
# for ((i = 0; i < length; i++)); do
#     datapath="${paths_list[i]}"
#     epochs="${epochs_list[i]}"

#     echo $datapath
#     echo $epochs

#     train_num_samples=$((epochs * 12800000))
#     #add 1 to i
#     epoch=$((i+1))

#     torchrun --master_port=12132 --nproc_per_node 8 train.py --scale medium_5x --data_dir $datapath --output_dir /home/pratyus2/logs/ --exp_name $exp_name --filter none --workers 10 --num_saves_per_epoch 10 --num_checkpoints $epoch --curriculum 1 --resume latest --train_num_samples $train_num_samples --total_steps 156250
#     # move all the checkpoints to a folder stepwise if they have the term "step"
#     mv /home/pratyus2/logs/$exp_name/checkpoints/*step* /home/pratyus2/logs/$exp_name/intermediate/
# done

declare -A path_epochs_dict

paths_list=("/drive2/datacomp_L14_0.5_to_0.6::/drive2/datacomp_L14_0.4_to_0.5::/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.4_to_0.5::/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.3_to_0.4::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1" "/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3::/drive2/datacomp_L14_0.0_to_0.1")
epochs_list=(18 15 8 9)
length=${#paths_list[@]}

exp_name=clipbucket_curriculum_adt_128M_5x
mkdir -p /home/pratyus2/logs/$exp_name/intermediate
# Loop through the lists and print the elements
for ((i = 0; i < length; i++)); do
    datapath="${paths_list[i]}"
    epochs="${epochs_list[i]}"
    
    echo $datapath
    echo $epochs
    
    train_num_samples=$((epochs * 12800000))
    #add 1 to i
    epoch=$((i+1))
    
    torchrun --master_port=12132 --nproc_per_node 1 train.py --scale medium_5x --data_dir $datapath --output_dir /home/pratyus2/logs/ --exp_name $exp_name --filter none --workers 10 --num_saves_per_epoch 10 --num_checkpoints $epoch --curriculum 1 --resume latest --train_num_samples $train_num_samples --total_steps 156250
    mv /home/pratyus2/logs/$exp_name/checkpoints/*step* /home/pratyus2/logs/$exp_name/intermediate/
done