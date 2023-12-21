
# cmlab30
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lib python train_llm_mc.py \
                                --epoch 100 \
                                --train_data_path data/train_data/train_civics_problem_database.json \
                                --valid_data_path data/train_data/valid_civics.json \
                                --batch_size 16 \
                                --accum_grad_step 1 \
                                --lr 2e-4 \
                                --lora_rank 16 \
                                --optimizer lion

# pc
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=lib python train.py \
#                                 --epoch 100 \
#                                 --train_data_path data/train_data/train_civics_problem_database.json \
#                                 --valid_data_path data/train_data/valid_civics.json \
#                                 --batch_size 16 \
#                                 --accum_grad_step 1 \
#                                 --lr 2e-4 \
#                                 --lora_rank 8 \
#                                 --with_answer_details
