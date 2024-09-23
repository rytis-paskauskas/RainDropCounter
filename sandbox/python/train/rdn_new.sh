#! /bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -A ICT24_CMSP
#SBATCH --gres=gpu:3
#SBATCH --output=rdn_new.log
#SBATCH --mem=0
#SBATCH --time=24:00:00
# module load profile/deeplrn
module load profile/candidate
module load cineca-ai/4.0.0
source /leonardo/home/userexternal/rpaskaus/venv/tensorflow/bin/activate
export NCCL_DEBUG=WARN
python 3_train_RDN.py train_new.ini
