#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2
# -m e
#$ -r y
#$ -N DA

conda activate ml

python3 ../DomainAdaptation.py --run_id=$SGE_TASK_ID --stage="all"

# qsub -t 200 DA.sh