CUDA_VISIBLE_DEVICES=0 ipython -- train_llm_mc.py \
                                --base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat \
                                --epoch 10 \
                                --train_data_path data/train_data/train_QB_history_9000.json \
                                --valid_data_path data/train_data/valid_QB_history_205.json \
                                --batch_size 16 \
                                --accum_grad_step 1 \
                                --lr 2e-4 \
                                --lora_rank 8 \
                                --optimizer lion