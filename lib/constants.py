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
    "step_by_step": "Think step by step.",
    "tips": "I will tip $200.",
}
