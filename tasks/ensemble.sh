#!/bin/bash
#$ -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2
#$ -m e
#$ -r y
#$ -N exp

conda activate ml
fsync $SGE_STDOUT_PATH &

python3 ../ensemble_training.py