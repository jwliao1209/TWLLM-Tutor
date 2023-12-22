
PYTHONPATH=lib python infer.py \
                --method lora-fine-tune \
                --test_data_path data/train_data/valid_history.json \
                --peft_path checkpoint_history/epoch=12_acc=0.5263157894736842 \
                --output_path prediction/lora_ep12_acc52_wo_108_num_beams_5.json
