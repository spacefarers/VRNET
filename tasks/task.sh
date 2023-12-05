#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2

#$ -l h=!(qa-p100-002|qa-p100-003)

# -m e
#$ -r y
#$ -N stnet_BL

conda activate ml

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=5 --finetune2_epochs=5 --cycles=50

# qsub -t 11-20 task.sh