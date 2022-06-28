#! /bin/bash
#PBS -N pretrain
#PBS -o pretrain.log
#PBS -e err.log
#PBS -l ncpus=2
#PBS -q cpu

module load compiler/anaconda3
cd /home/kuber.shahi_asp22/project
source aml/bin/activate
rm pretrain.log
rm err.log
python pretrain.py