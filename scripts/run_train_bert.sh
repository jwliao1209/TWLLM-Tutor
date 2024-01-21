#!/bin/bash

TRAIN_DATA=${TRAIN_DATA:-GSAT_social}    # {GSAT, QB}_{social, history, civics}
VALID_DATA=${VALID_DATA:-GSAT_social}  # {GSAT, QB}_{social, history, civics}

CUDA_VISIBLE_DEVICES=0 python train.py \
                    --config_path configs/bert-multiple_choice-train_${TRAIN_DATA}-valid_${VALID_DATA}-wo_answer_details.yaml
