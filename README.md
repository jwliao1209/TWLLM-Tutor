# Taiwan LLM Tutor: Language Models for Taiwanese Education

## Set the Environment

### Conda

```bash
conda create --name adl_final python=3.10.0
source activate adl_final
pip install -r requirements.txt
```

### Virtual Environment

```bash
virtualenv --python=python3.10 adl_final
source ~/adl_final/bin/activate
pip install -r requirements.txt
```

### Pyenv

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 adl_final
pip install -r requirements.txt
```

## Dataset

The GSAT social dataset is downloaded from [GSAT Website](https://www.ceec.edu.tw/files/file_pool/1/0j076464103640279375/04-105%e5%ad%b8%e6%b8%ac%e7%a4%be%e6%9c%83%e7%ad%94%e6%a1%88.pdf).

### File Structure

```
data
|- GSAT_social  # Public is for data we consider publishable (without copyright issues, etc).
|- QB_social    # Private is for data we don't want to publish (For future extensions).
```

### Dataset for Chinese BERT
The data stored in `data/train_data/GSAT_social_with_image` has been preprocessed using the following commands:

```bash
python prepare_data/convert_vision_mc_format.py
python prepare_data/prepare_embeddings.py
```

### Dataset for Taiwan LLM
#### Data Format
```
{
        "subject": "social_study",
        "year": "83",
        "id": 1,
        "type": "single",
        "question": "孫中山先生認為造成中國人像一盤散沙，民族不夠團結的主因為何",
        "A": "任外族帝制專斷的統治下，人民喪失了關心公共事務的能力",
        "B": "異族的征服者過於強大，中國人團結也沒用",
        "C": "中國入的家族觀念過於發達",
        "D": "過早提倡天下一家的世界主義",
        "answer": "A",
        "answer_details": ""
},
```

## Training

### Chinese BERT Multiple Choice

To launch the experiments involving BERT and Vision-BERT, use the following command:

```bash
    sh scripts/launch_mc_experiments.sh
```

### Taiwan LLM Multiple Choice with QLoRA
To fine-tune the Taiwan LLM with multiple choice and QLoRA, you can run the command:
```bash
bash scripts/train_twllm_qlora_mc.sh
```

### Taiwan LLM Instruction Tuning with QLoRA
To fine-tune the Taiwan LLM with instruction tuning and QLoRA, you can run the command:
```bash
bash scripts/train_twllm_qlora_it.sh
```

### Taiwan LLM Instruction Tuning with LoftQ
We modified the [quantize_save_load.py](quantize_save_load.py) to apply LoftQ for TWLLM model.

Below is an example of obtaining 4bit LLAMA-2-7b with 16-rank LoRA adapters by 5 alternating steps.

```bash
SAVE_DIR="model_weight/"
python quantize_save_load.py \
    --model_name_or_path model_weight/Taiwan-LLM-7B-v2.0-chat \
    --token model_weight/Taiwan-LLM-7B-v2.0-chat \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR
```

#### Model Setting

Similar to loading from Huggingface Hub, we only need to change the `MODEL_ID` to the `MODEL_DIR`.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "model_weight/Taiwan-LLM-7B-v2.0-chat-4bit-16rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    subfolder="loftq_init",
    is_trainable=True,
)
```

The above commands end up with creating the model directory under `$SAVE_DIR`.
Specifically, the model directory is named as

`MODEL_DIR = SAVE_DIR + f"{args.model_name_or_path.split('/')[-1]}-{args.bits}bits-{args.rank}rank"`

In this example, `MODEL_DIR="model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"`, where the backbone is stored in `$MODEL_DIR`
and the LoRA adapters are at the sub-folder `$MODEL_DIR/loftq_init`.


To fine-tune the Taiwan LLM withinstruction tuning and LoftQ, you can run the command:
```bash
bash scripts/train_twllm_loftq_it.sh
```

### Taiwan LLM Multiple Choice with LoftQ

```bash
SAVE_DIR="model_weight/"
python quantize_save_load.py \
    --model_name_or_path model_weight/Taiwan-LLM-7B-v2.0-chat \
    --token model_weight/Taiwan-LLM-7B-v2.0-chat \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR \
	--mc_model
```

To fine-tune the Taiwan LLM with multiple choice and LoftQ, you can run the command:
```bash
bash scripts/train_twllm_loftq_mc.sh
```

## Debug

You can use --pdb for debugging.

```bash
ipython --pdb -- {Python File}.py
```


## Inference

```bash
ipython -- twllm/infer_twllm_qlora_it.py
```

## OS and Hardware

The experiments were performed on a personal computer equipped with a single NVIDIA GeForce RTX 4090 GPU with 24 GB of VRAM, and a server configuration featuring a single RTX A6000 GPU with 49 GB of VRAM.


## Acknowledgement
We thank the Taiwan LLM repository: https://github.com/MiuLab/Taiwan-LLaMa


## Citation
```bibtex
@misc{
    title  = {TWLLM Tutor: Revolutionizing Taiwanese Secondary Education with Large Language Model},
    author = {Jia-Wei Liao, Ji-Jia Wu, Kun-Hsiang Lin, Kang-Yang Huang},
    url    = {https://github.com/jwliao1209/TWLLM-Tutor},
    year   = {2023}
}
```
