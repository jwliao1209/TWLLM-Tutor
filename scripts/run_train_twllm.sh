#!/bin/bash

FINETUNE_METHOD=${FINETUNE_METHOD:-qlora_instruction_tuning}  # {qlora, loftq}_{instruction_tuning, multiple_choice}
TRAIN_DATA=${TRAIN_DATA:-QB_history}                          # {GSAT, QB}_{social, history, civics}
VALID_DATA=${VALID_DATA:-GSAT_history}                        # {GSAT, QB}_{social, history, civics}
ANSWER_DETAILS=${ANSWER_DETAILS:-w_answer_details}            # w_answer_details, wo_answer_details

CUDA_VISIBLE_DEVICES=0 python train.py \
                    --config_path configs/twllm-${FINETUNE_METHOD}-train_${TRAIN_DATA}-valid_${VALID_DATA}-${ANSWER_DETAILS}.yaml
