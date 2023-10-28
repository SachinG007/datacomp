datalist=(0.2 0.3 0.1 0.2 0.3 0.1 0.2 0.4 0.1 0.3 0.2 0.4 0.1 0.3 0.2 0.4 0.5 0.1 0.3 0.2 0.4 0.5 0.1 0.3 0.2 0.1 0.4 0.5 0.3 0.6 0.1 0.2 0.5 0.4 0.3 0.6 0.1 0.2 0.3 0.5 0.4 0.1 0.6 0.3 0.5 0.2 0.4 0.1 0.6 0.3)


exp_name=clipbucket_curriculum_128M_5x
mkdir -p /home/pratyus2/logs/$exp_name/intermediate

for i in "${datalist[@]}"
do
    echo $i
    start=$(printf "%.1f" $(echo "$i - 0.1" | bc))
    
    end=$(printf "%.1f" $i)
    var="datacomp_L14_${start}_to_${end}"
    
    echo $var
    
    datapath="/drive2/$var"
    
    torchrun --master_port=12132 --nproc_per_node 8 train.py --scale medium_5x --data_dir $datapath --output_dir /home/pratyus2/logs/ --exp_name $exp_name --filter none --workers 10 --num_saves_per_epoch 1 --num_checkpoints 50 --curriculum 1 --resume latest
    
    # move all the checkpoints to a folder stepwise if they have the term "step"
    mv /home/pratyus2/logs/$exp_name/checkpoints/*step* /home/pratyus2/logs/$exp_name/intermediate/
done