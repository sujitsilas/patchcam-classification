#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.metastatic_tissue_classification_inception

## Edit the line below as needed:
# GPU Model             Compute Capability   CUDA Cores   Memory
# A100                  4                    6912         80 GB   A100
# Tesla V100            1                    5120         32 GB   V100
# GeForce RTX 2080 Ti   2                    4352         10 GB   RTX2080Ti
# Tesla P4              1                    2560         8 GB    P4

#$ -l gpu,A100,cuda=1,h_rt=8:00:00 

# Email address to notify
#$ -M $sujitsilas@g.ucla.edu

# Notify when
#$ -m bea

# Load the required modules
cd /u/scratch/s/sujit009/metastatic_tissue_classification/

export TORCH_HOME=/u/scratch/s/sujit009/metastatic_tissue_classification/

. /etc/bashrc
module load python/3.7.3
module load cuda/11.8
module load python
export PATH="{PATH}:~/.local/bin"

# Activate virtual env
source /u/scratch/s/sujit009/metastatic_tissue_classification/myenv/bin/activate

# Run your training script with Python
python main_inception.py

# Deactivate virtual environment upon completion
deactivate