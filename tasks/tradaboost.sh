#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-v100-* -l gpu=2

# -l h=!(qa-p100-002|qa-p100-003)

# -m e
#$ -r y
#$ -N tradaboost

conda activate ml

python3 ../TrAdaboost.py --run_id=$SGE_TASK_ID --boosting_iters=1 --cycles=45