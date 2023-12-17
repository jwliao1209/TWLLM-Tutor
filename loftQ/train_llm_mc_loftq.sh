# pip install -r LoftQ/requirments.txt

PYTHONPATH=lib ipython --pdb -- train_llm_mc_loftq.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat-4bit-16rank-mc \
	--epoch 100 \
	--train_data_path data/train_data/train_history_problem_database.json \
	--valid_data_path data/train_data/valid_history.json \
	--batch_size 2 \
	--accum_grad_step 4 \
	--lora_rank 16 \
	--lr 2e-4
