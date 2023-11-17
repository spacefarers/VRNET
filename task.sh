#!/bin/bash
#$ -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2
#$ -m e
#$ -r y
#$ -N exp

conda activate ml

# pretrain and finetune

python3 repeat_test.py
