datalist = (0.2 0.3 0.1 0.2 0.3 0.2 0.1 0.4 0.2)

exp_name=curriculum
mkdir /home/pratyus2/logs/$exp_name/intermediate

for i in "${datalist[@]}"
do
    echo $i
    datapath="/drive/datacomp/shards/datacomp_clipL14_medium_"$i".pt"
    
    torchrun --master_port=12332 --nproc_per_node 4 train.py --scale medium --data_dir $datapath --output_dir /home/pratyus2/logs/ --exp_name $exp_name --filter none --workers 10 --num_saves_per_epoch 2 --num_checkpoints 10 --curriculum 1 --resume latest
    
    # move all the checkpoints to a folder stepwise if they have the term "step"
    mv /home/pratyus2/logs/clipbucket_top10p/*step* /home/pratyus2/logs/clipbucket_top10p/intermediate/
    
done