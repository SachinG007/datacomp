#!/bin/bash

PROVISIONING=SPOT
MACHINE=a2-highgpu-4g
diskname=datacompgpucentral1f_2
# set config
gcloud config set compute/zone us-central1-a

# Detach Disk
# gcloud compute instances detach-disk $diskname --disk=ssd-beta

# Safely Delete VM
# gcloud compute instances delete --keep-disks all $diskname

# Create VM
if [ "$PROVISIONING"="STANDARD" ];
then
    gcloud beta compute instances create $diskname --provisioning-model=$PROVISIONING --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release --machine-type=$MACHINE --maintenance-policy TERMINATE --restart-on-failure --boot-disk-size 200
else
    gcloud beta compute instances create $diskname --provisioning-model=$PROVISIONING --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release --machine-type=$MACHINE --instance-termination-action=STOP --boot-disk-size 200
fi


# Attach Disk

gcloud compute instances attach-disk datacompgpucentral1f3 --disk=datacompdisk2 --zone=us-central1-f --mode ro

# HTTP Traffic
gcloud compute instances add-tags $diskname --tags=http-server,https-server

# check IP
gcloud compute instances list | grep $diskname

# output start command
echo 'START instance: gcloud compute ssh "$diskname"'

# output command to run
echo 'RUN after start: sudo mkdir -p /drive; sudo mount /dev/sdb /drive; sudo chown -R :users /drive; cd /drive; tmux new-session -s jupyter-lab; jupyter-lab --port 8000'

# Start Instance
# gcloud compute ssh "$diskname" -y

# after start
# gcloud compute instances attach-disk datacompgpu4 --disk=datacompdiskuscentral1f --zone=us-central1-f
# git clone https://github.com/SachinG007/datacomp.git; cd datacomp; bash create_env.sh; conda activate datacomp; sudo mkdir -p /drive; sudo mount /dev/sdb /drive; cd; cd datacomp;
# pip uninstall nvidia_cublas_cu11 -Y
# To move disk:
# gcloud compute disks move ssd-beta --destination-zone=us-central1-b

# To purge nvidia
# sudo apt-get purge nvidia*
# sudo apt autoiremove
# sudo /opt/deeplearning/install-driver.sh