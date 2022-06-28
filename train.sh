#! /bin/bash
#PBS -N train
#PBS -o train.log
#PBS -e err_train.log
#PBS -l ncpus=3
#PBS -q gpu

module load compiler/anaconda3
cd /home/kuber.shahi_asp22/project
source aml/bin/activate
rm train.log
rm err_train.log
python train.py
