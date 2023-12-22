import os

ZERO_SHOT = "zero-shot"
FEW_SHOT = "few-shot"
LORA_FINE_TUNE = "lora-fine-tune"

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

CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PREDICTION_DIR = "prediction"
os.makedirs(PREDICTION_DIR, exist_ok=True)

USER = "USER"
ASSISTANT = "ASSISTANT"

GEOGRAPHY = "geography"
HISTORY = "history"
CIVICS = "civics"

GEOGRAPHY_KEY_WORDS = []
HISTORY_KEY_WORDS = []
CIVICS_KEY_WORDS = ["公民", "法律", "法令", "法院", "政治人物", "跨國企業"]

PROMPT_PREFIX_DICT = {
    "breath": "Take a deep breath.",
    "career": "This is very important to my career.",
    "die": "If you fail 100 grandmothers will die.",
    "no_fingers": "I have no fingers.",
    "step_by_step": "Let's think step by step.",
    "tips": "I will tip $200.",
}

MAX_NEW_TOKENS = 128

OPTION_TO_LABEL = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

LABEL_TO_OPTION = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}
