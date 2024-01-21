import os


PROJECT_NAME = "adl_final_project"

CONFIGS_DIR = "configs"
CHECKPOINT_DIR = "checkpoint"

GSAT_SOCIAL_FOLDERS = "data/train_data/GSAT_social"
TRAIN_GSAT_SOCIAL = os.path.join(GSAT_SOCIAL_FOLDERS, "train_GSAT_social.json")
VALID_GSAT_CIVICS = os.path.join(GSAT_SOCIAL_FOLDERS, "valid_GSAT_civics.json")
VALID_GSAT_HISTORY = os.path.join(GSAT_SOCIAL_FOLDERS, "valid_GSAT_history.json")
VALID_GSAT_SOCIAL = os.path.join(GSAT_SOCIAL_FOLDERS, "valid_GSAT_social.json")

QB_SOCIAL_FOLDERS = "data/train_data/QB_social"
TRAIN_QB_CIVICS = os.path.join(QB_SOCIAL_FOLDERS, "train_QB_civics.json")
TRAIN_QB_GEOGRAPHY = os.path.join(QB_SOCIAL_FOLDERS, "train_QB_geography.json")
TRAIN_QB_HISTORY = os.path.join(QB_SOCIAL_FOLDERS, "train_QB_history.json")
TRAIN_QB_SOCIAL = os.path.join(QB_SOCIAL_FOLDERS, "train_QB_social.json")
VALID_QB_HISTORY = os.path.join(QB_SOCIAL_FOLDERS, "valid_QB_history.json")


PREDICTION_FILE = "prediction.json"
CONFIG_FILE = "config.yaml"

TWLLM = "twllm"
BERT = "bert"

ZERO_SHOT = "zero_shot"
FEW_SHOT = "few_shot"

QLORA = "qlora"
LOFTQ = "loftq"

INSTRUCTION_TUNING = "instruction_tuning"
MULTIPLE_CHOICE = "multiple_choice"

TRAIN_FOLDERS = [
    "83",
    "84",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "91",
    "91_bu",
    "92",
    "92_bu",
    "93",
    "94",
    "95",
    "96",
    "97",
    "98",
    "99",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
]

VALID_FOLDERS = [
    "108",
    "109",
    "110",
    "111",
    "112",
]

USER = "USER"
ASSISTANT = "ASSISTANT"

GEOGRAPHY = "geography"
HISTORY = "history"
CIVICS = "civics"

OPTION = ["A", "B", "C", "D"]
LABEL = [0, 1, 2, 3]
OPTION_TO_LABEL = dict(zip(OPTION, LABEL))
LABEL_TO_OPTION = dict(zip(LABEL, OPTION))

PROMPT_PREFIX_DICT = {
    "breath": "Take a deep breath.",
    "career": "This is very important to my career.",
    "die": "If you fail 100 grandmothers will die.",
    "no_fingers": "I have no fingers.",
    "step_by_step": "Let's think step by step.",
    "tips": "I will tip $200.",
}
