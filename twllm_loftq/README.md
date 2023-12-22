# LoftQ DIY

## Apply LoftQ and save

I modified the [quantize_save_load.py](quantize_save_load.py) to apply LoftQ for TWLLM model.

Below is an example of obtaining 4bit LLAMA-2-7b with 16-rank LoRA adapters by 5 alternating steps.

### Cuasal language modeling

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
