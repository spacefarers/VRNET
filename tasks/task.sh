#!/bin/bash
#$ -M michaelx@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2
#$ -m e
#$ -r y
#$ -N exp

conda activate ml

# pretrain and finetune

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=50 --finetune2_epochs=200

# qsub -t 11-20 task.sh