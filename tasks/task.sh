#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2
# -m e
#$ -r y
#$ -N stnet_BL

conda activate ml

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=400 --finetune2_epochs=50

# qsub -t 1-10 task.sh