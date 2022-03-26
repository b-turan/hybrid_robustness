#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=berkant.ferhat.turan@hhi.fraunhofer.de
#SBATCH --job-name=supervised_hdgm_cifar10_csv_logging_alpha_1e0_test
#SBATCH --output=runs/%j_%x.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#xxxSBATCH --time=2-0:00:00


# After this [on the cluster] <-> [in the container]
#                 $LOCAL_DATA  = /data
#              $LOCAL_JOB_DIR  = /mnt/output
#                       ./code = /opt/code

#create singularity container via singularity build --force --fakeroot base_pytorch.sif base_pytorch.def


source "/etc/slurm/local_job_dir.sh"

#############################################
#print environmental variables
echo "====================================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "USER: $USER"

#DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S'`}
#echo "DATESTAMP: $DATESTAMP"
DATE=`date`
echo "DATE: $DATE"
#OUTPUTFOLDER="$SLURM_JOB_ID""_""$SLURM_JOB_NAME"
OUTPUTFOLDER="$SLURM_JOB_ID"
echo "OUTPUTFOLDER: $OUTPUTFOLDER"
#printenv
GIT_COMMIT=`git rev-parse --short HEAD`
echo "GIT_COMMIT: $GIT_COMMIT"
#metadata passed to executable
metadata="SLURM_JOB_ID=$SLURM_JOB_ID;SLURM_JOB_NAME=$SLURM_JOB_NAME;SLURMD_NODENAME=$SLURMD_NODENAME;GIT_COMMIT=$GIT_COMMIT;USER=$USER"
#echo $metadata
echo "====================================================="

#############################################################################################
#set output directories

#OUTPUTPATH_JOB="/opt/submit/runs/$OUTPUTFOLDER"
OUTPUTPATH_JOB="/opt/output"
SUBMIT_DIR=`pwd`
OUTPUTPATH_LOCAL="$SUBMIT_DIR/runs/$OUTPUTFOLDER"
#create temporary output directory
mkdir -p "${LOCAL_JOB_DIR}/job_results"

#############################################################################################
#SET DATASET
#dataset="cifar10"
dataset="svhn"
#############################################################################################
SINGULARITY_BINDPATH="./code:/opt/code,./:/opt/submit,${LOCAL_JOB_DIR}/job_results:/opt/output"

if [ $dataset = "svhn" ]
then

    datasetfolder1="svhn"
    cd $LOCAL_JOB_DIR
    mkdir data
    cp -r $HOME/datasets/$datasetfolder1 data

    datapaths="--data /data/$datasetfolder1"
    export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH},${LOCAL_JOB_DIR}/data:/data/"
elif [ $dataset = "cifar10" ]
then
    datapaths="--data /data"
    export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH},${LOCAL_DATA}/datasets/cifar-10-batches-py:/data/cifar-10-batches-py"
fi
cd $SUBMIT_DIR
# data is now located at /data/svhn and /data/cifar-10-batches-py respectively

#################################################################################################
# all your commandline arguments- data path should be passed via --data and output should be written to --output_path
cmd="python /opt/code/test_model.py --gpus 1 --dataset cifar10 --ood_cifar100 True --eval True --histogram True --fgsm True --eps_interval 0.02 0.08 --seed True --checkpoint_path /data/svhn/checkpoints/alpha_1e0/460157/csv_logs/version_0/checkpoints/epoch=130-step=13754.ckpt"


#################################################################################################
cmd="$cmd $datapaths --output_path $OUTPUTPATH_JOB"

echo "Command: $cmd"
singularity exec --nv base_pytorch.sif $cmd


#copying results from local 
mkdir -p $OUTPUTPATH_LOCAL
cp -r ${LOCAL_JOB_DIR}/job_results/* $OUTPUTPATH_LOCAL
rm -r ${LOCAL_JOB_DIR}/job_results
#also copy output
cp "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out" "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"

