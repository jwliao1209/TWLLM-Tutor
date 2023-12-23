NBIT=4
LORA_RANK=16
TASK_TYPE=IT

ipython --pdb -- twllm_loftq/quantize_save_load.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat \
	--token model_weight/Taiwan-LLM-7B-v2.0-chat \
	--nbit $NBIT \
	--lora_rank $LORA_RANK \
	--task_type $TASK_TYPE

ipython --pdb -- twllm_loftq/train_twllm_loftq_it.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat-${NBIT}bit-${LORA_RANK}rank-$TASK_TYPE \
	--epoch 10 \
	--train_data_path data/train_data/train_QB_history_9000.json \
	--valid_data_path data/train_data/valid_QB_history_205.json \
	--batch_size 4 \
	--accum_grad_step 4 \
	--nbit $NBIT \
	--lora_rank $LORA_RANK \
	--lr_scheduler cosine \
	--warm_up_step 300 \
	--with_answer_details \
	--lr 2e-4
