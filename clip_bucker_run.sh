echo "running "
torchrun --master_port=12332 --nproc_per_node 8 train.py --scale medium --data_dir /drive2/datacomp/shards/ --output_dir logs --exp_name clipbucket_30p_to_40p --filter is_valid --valid_file datacomp_clipL14_medium_0.3_to_0.4.pt --workers 10 --num_saves_per_epoch 20
echo "running "
torchrun --master_port=12332 --nproc_per_node 8 train.py --scale medium --data_dir /drive2/datacomp/shards/ --output_dir logs --exp_name clipbucket_50p_to_60p --filter is_valid --valid_file datacomp_clipL14_medium_0.5_to_0.6.pt --workers 10 --num_saves_per_epoch 20
echo "running "
torchrun --master_port=12332 --nproc_per_node 8 train.py --scale medium --data_dir /drive2/datacomp/shards/ --output_dir logs --exp_name clipbucket_60p_to_bottom --filter is_valid --valid_file datacomp_clipL14_medium_0.6_to_bottom.pt --workers 10 --num_saves_per_epoch 20