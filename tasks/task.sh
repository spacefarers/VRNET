#!/bin/bash
#$ -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2
#$ -m e
#$ -r y
#$ -N exp

conda activate ml
fsync $SGE_STDOUT_PATH &

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=50 --finetune2_epochs=200

# qsub -t 1-10 task.sh