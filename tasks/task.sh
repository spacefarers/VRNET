#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2

# -l h=!(qa-p100-002|qa-p100-003)

# -m e
#$ -r y
#$ -N stnet_BL

conda activate ml

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=20 --finetune2_epochs=0 --cycles=20 --swap_source_target=False --use_all_data=False

# qsub -t 006-010 task.sh