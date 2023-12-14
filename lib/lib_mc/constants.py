import os

DATA_DIR = "data/train_data_mc"
CHECKPOINT_DIR = "checkpoint"
PREDICTION_DIR = "pred"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

MC_TRAIN_FILE = "train.json"
MC_VALID_FILE = "valid.json"
MC_DATA_FILE = {
    "train": os.path.join(DATA_DIR, MC_TRAIN_FILE),
    "valid": os.path.join(DATA_DIR, MC_VALID_FILE),
}
MC_ENDING_LEN = 4
MC_LAB_COL_NAME = "label"
MC_MAX_SEQ_LEN = 512
