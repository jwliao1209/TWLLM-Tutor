TASK_TYPE=instruction_tuning
LORA_RANK=16
NBIT=4

CUDA_VISIBLE_DEVICES=0 python quantize_twllm_loftq.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat \
	--token model_weight/Taiwan-LLM-7B-v2.0-chat \
    --task_type $TASK_TYPE \
	--lora_rank $LORA_RANK \
    --nbit $NBIT
