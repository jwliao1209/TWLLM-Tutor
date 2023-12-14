ipython --pdb -- prepare_data/convert_train_mc_format.py

ipython --pdb -- train_mc.py \
	--train_data data/train_database_mc/train.json \
	--valid_data data/train_database_mc/valid.json \
	--tokenizer_name kevinzyz/chinese-roberta-wwm-ext-finetuned-MC-hyper \
	--model_name_or_path kevinzyz/chinese-roberta-wwm-ext-finetuned-MC-hyper \
	--batch_size 8 \
	--accum_grad_step 4 \
	--lr 0.00002 \
	--epoch 10 \
	--bf16