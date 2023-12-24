CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --exp_name bert-exp1 \
    --use_train_gsat_83_107 \
    --use_valid_gsat_all

CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --exp_name bert-exp2 \
    --use_train_qb_history \
    --use_valid_qb_history

CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --exp_name bert-exp3 \
    --use_train_qb_history \
    --use_valid_gsat_history

CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --exp_name bert-exp4 \
    --use_train_qb_civics \
    --use_valid_gsat_civics

CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --exp_name bert-exp5 \
    --use_train_qb_civics \
    --use_train_qb_history \
    --use_valid_gsat_all
