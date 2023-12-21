# pip install -r LoftQ/requirments.txt

PYTHONPATH=lib ipython --pdb -- loftQ/train_llm_mc_loftq.py \
	--base_model_path model_weight/Taiwan-LLM-7B-v2.0-chat-2bit-8rank-mc \
	--epoch 10 \
	--train_data_path data/train_data/train_history_problem_database.json \
	--valid_data_path data/train_data/valid_history.json \
	--batch_size 2 \
	--accum_grad_step 4 \
	--lora_rank 8 \
	--lr 3e-5
