ipython --pdb -- prepare_data/convert_train_mc_format.py

ipython --pdb -- train_mc.py \
	--tokenizer_name kevinzyz/chinese-roberta-wwm-ext-finetuned-MC-hyper \
	--model_name_or_path kevinzyz/chinese-roberta-wwm-ext-finetuned-MC-hyper \
	--batch_size 8 \
	--accum_grad_step 4 \
	--epoch 10 \
	--weight_decay 0.001 \
	--bf16