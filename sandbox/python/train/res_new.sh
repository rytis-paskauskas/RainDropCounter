#! /bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -A ICT24_CMSP
#SBATCH --gres=gpu:3
#SBATCH --output=res_new.log
#SBATCH --time=24:00:00
#SBATCH --mem=0
# module load profile/deeplrn
module load profile/candidate
module load cineca-ai/4.0.0
source /leonardo/home/userexternal/rpaskaus/venv/tensorflow/bin/activate
export NCCL_DEBUG=WARN
python 3_train_residual.py train_new.ini
