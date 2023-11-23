#!/bin/bash
#$ -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2
#$ -m e
#$ -r y
#$ -N ensemble

conda activate ml

python3 ../ensemble_training.py