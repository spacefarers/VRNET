#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2
# -m e
#$ -r y
#$ -N ensemble

conda activate ml

python3 ../ensemble_training.py --run_id=$SGE_TASK_ID --finetune1_epochs=5 --finetune2_epochs=5 --ensemble_epochs=15

# qsub -t 110-119 ensemble.sh