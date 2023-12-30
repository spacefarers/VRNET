#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2
# -m e
#$ -r y
#$ -N ensemble

conda activate ml

python3 ../DomainAdaptation.py --run_id=$SGE_TASK_ID --finetune1_epochs=20 --finetune2_epochs=0

# qsub -t 200 DA.sh