
# cmlab30
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=lib python train_llm_mc.py \
                                --epoch 100 \
                                --train_data_path data/train_data/train_QB_civics_2035.json \
                                --valid_data_path data/train_data/valid_GSAT_civics-108-112_79.json \
                                --batch_size 4 \
                                --accum_grad_step 1 \
                                --lr 2e-4 \
                                --lora_rank 16 \

# pc
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=lib python train.py \
#                                 --epoch 100 \
#                                 --train_data_path data/train_data/train_QB_civics_2035.json \
#                                 --valid_data_path data/train_data/valid_GSAT_civics-108-112_79.json \
#                                 --batch_size 16 \
#                                 --accum_grad_step 1 \
#                                 --lr 2e-4 \
#                                 --lora_rank 8 \
#                                 --with_answer_details \
#                                 --optimizer lion
