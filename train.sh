# pip install -r LoftQ/requirments.txt

PYTHONPATH=lib python train.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat \
	--epoch 100 \
	--train_data_path data/train_data/train_history_problem_database.json \
	--valid_data_path data/train_data/valid_history.json \
	--batch_size 16 \
	--accum_grad_step 1 \
	--lr 1e-4 \
	--lora_rank 16