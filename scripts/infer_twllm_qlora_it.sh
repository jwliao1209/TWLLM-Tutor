ipython -- twllm_qlora/infer_twllm_qlora_it.py \
            --method lora-fine-tune \
            --test_data_path data/train_data/train_QB_history_9000.json \
            --peft_path checkpoint_history/epoch=12_acc=0.5263157894736842 \
            --output_path prediction/lora_ep12_acc52_wo_108.json
