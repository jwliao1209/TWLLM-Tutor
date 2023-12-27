# TWLLM Tutor: Revolutionizing Taiwanese Secondary Education with Large Language Model

## Set the Environment

### Conda

```bash
conda create --name adl_final python=3.10.0
source activate adl_final
pip install -r requirements.txt
```

### Pyenv

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 adl_final
pip install -r requirements.txt
```

### Virtual Environment

```bash
virtualenv --python=python3.10 adl_final
source ~/adl_final/bin/activate
pip install -r requirements.txt
```


## Prepare Training Dataset

The data is stored in the train_data variable with the following naming convention: `{train, valid}_{GSAT, QB}_{subject with years}_{number of questions}`.

```bash
ipython prepare_data/convert_to_train_format.py
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 ipython -- train.py \
                                --epoch 100 \
                                --batch_size 32
```


## Inference

```bash
ipython -- infer.py
```

## Debug

You can use --pdb for debugging.

```bash
ipython --pdb -- train.py \
                --epoch 100 \
                --batch_size 32
```
Here's a revised version of your text with improved clarity:


## Multiple Choice with BERT

The data stored in `data/train_data/GSAT_social_with_image` has been preprocessed using the following commands:

```bash
python prepare_data/convert_vision_mc_format.py
python prepare_data/prepare_embeddings.py
```

To launch the experiments involving BERT and Vision-BERT, use the following command:

```bash
    sh scripts/launch_mc_experiments.sh
```

## LoftQ
## Apply LoftQ and save

I modified the [quantize_save_load.py](quantize_save_load.py) to apply LoftQ for TWLLM model.

Below is an example of obtaining 4bit LLAMA-2-7b with 16-rank LoRA adapters by 5 alternating steps.

### Causal language modeling

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

The above commands end up with creating the model directory under `$SAVE_DIR`.
Specifically, the model directory is named as

`MODEL_DIR = SAVE_DIR + f"{args.model_name_or_path.split('/')[-1]}-{args.bits}bits-{args.rank}rank"`

In this example, `MODEL_DIR="model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"`, where the backbone is stored in `$MODEL_DIR`
and the LoRA adapters are at the sub-folder `$MODEL_DIR/loftq_init`.

### Multiple choice

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

### Load and train

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

### Train cuasal language modeling

```bash
bash train_llm_loftq.sh
```

### Train multiple choice

```bash
bash train_llm_mc_loftq.sh
```

## OS and Hardware

We implemented the code on an environment running Ubuntu 22.04.3, utilizing a 12th Gen Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB VRAM.


## Acknowledgement
We thank the Taiwan-LLaMa repository: https://github.com/MiuLab/Taiwan-LLaMa


## Reference

-   https://www.ceec.edu.tw/files/file_pool/1/0j076464103640279375/04-105%e5%ad%b8%e6%b8%ac%e7%a4%be%e6%9c%83%e7%ad%94%e6%a1%88.pdf


## Citation
```bibtex
@misc{
    title  = {TWLLM Tutor: Revolutionizing Taiwanese Secondary Education with Large Language Model},
    author = {Jia-Wei Liao, Ji-Jia Wu, Kun-Hsiang Lin, Kang-Yang Huang},
    url    = {https://github.com/jwliao1209/TWLLM-Tutor},
    year   = {2023}
}
```
