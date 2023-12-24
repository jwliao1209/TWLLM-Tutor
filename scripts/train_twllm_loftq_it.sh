NBIT=4
LORA_RANK=16
TASK_TYPE=IT

# ipython --pdb -- twllm_loftq/quantize_save_load.py \
# 	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat \
# 	--token model_weight/Taiwan-LLM-7B-v2.0-chat \
# 	--nbit $NBIT \
# 	--lora_rank $LORA_RANK \
# 	--task_type $TASK_TYPE

CUDA_VISIBLE_DEVICES=2 ipython --pdb -- twllm_loftq/train_twllm_loftq_it.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat-${NBIT}bit-${LORA_RANK}rank-$TASK_TYPE \
	--results_dir results \
	--epoch 10 \
	--train_data_path data/train_data/train_QB_history_9000.json \
	--valid_data_path data/train_data/valid_GSAT_history-108-112_97.json \
	--batch_size 4 \
	--accum_grad_step 4 \
	--nbit $NBIT \
	--lora_rank $LORA_RANK \
	--with_answer_details \
	--lr 2e-4
