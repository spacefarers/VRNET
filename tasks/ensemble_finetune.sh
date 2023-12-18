#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2

fsync $SGE_STDOUT_PATH &

# -m e
#$ -r y
#$ -N EN-FT

conda activate ml

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=5 --finetune2_epochs=5 --cycles=5 --load_ensemble_model=True --tag=EN-FT

# qsub -t 210-219 ensemble_finetune.sh