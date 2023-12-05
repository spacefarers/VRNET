#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=2

# temporarily exclude qa-p100-002 and 3
#$ -l h=!(qa-p100-002|qa-p100-003)


# -m e
#$ -r y
#$ -N EN-FT

conda activate ml

python3 ../run.py --run_id=$SGE_TASK_ID --finetune1_epochs=5 --finetune2_epochs=5 --cycles=5 --load_ensemble_model=True --tag=EN-FT

# qsub -t 201-210 ensemble_finetune.sh