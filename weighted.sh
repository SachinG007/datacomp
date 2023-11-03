exp_name=clipbucket_64M_top30
datapath="/drive2/datacomp_L14_0.0_to_0.1::/drive2/datacomp_L14_0.1_to_0.2::/drive2/datacomp_L14_0.2_to_0.3"
# weights="0.11779862055688131::0.14059484712967021::0.11882185772520693::0.03595843"

torchrun --master_port=12532 --nproc_per_node 4 train.py --scale tmars_64 --data_dir $datapath --output_dir /home/pratyus2/logs/ --exp_name $exp_name --filter none --workers 10 --num_saves_per_epoch 10 --num_checkpoints 1