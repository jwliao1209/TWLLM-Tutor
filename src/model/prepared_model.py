from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMultipleChoice

from ..configs import get_bnb_config
from ..constants import TWLLM, BERT
from ..constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE
from ..constants import QLORA, LOFTQ, ZERO_SHOT, FEW_SHOT
from ..constants import OPTION


def get_model(
    model_type: str,
    base_model_path: str,
    finetune_type: str,
    adapter: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    device = None,
    tokenizer: Optional[AutoTokenizer] = None,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[int] = None,
    is_trainable: bool = True,
):
    """
    A function for prepared the model (twllm or bert).
    """

    bnb_config = get_bnb_config()

    if finetune_type in [ZERO_SHOT, FEW_SHOT]:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )

    elif finetune_type == INSTRUCTION_TUNING:
        if adapter == QLORA:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            if is_trainable:
                model = prepare_model_for_kbit_training(model)
                peft_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
            else:
                model = PeftModel.from_pretrained(model, checkpoint_path)

        elif adapter == LOFTQ:
            # ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning#load-and-train
            bnb_config.bnb_4bit_use_double_quant = False
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
            model = PeftModel.from_pretrained(
                model,
                base_model_path,
                subfolder="loftq_init",
                is_trainable=is_trainable,
            )
            if is_trainable:
                model.print_trainable_parameters()

        else:
            raise ValueError(f"Unsupported adapter type: {adapter}")

    elif finetune_type == MULTIPLE_CHOICE:
        if model_type == TWLLM:
            if adapter == QLORA:
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=len(OPTION),
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                )
                if is_trainable:
                    model = prepare_model_for_kbit_training(model)
                    peft_config = LoraConfig(
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.SEQ_CLS,
                    )
                    model = get_peft_model(model, peft_config)
                else:
                    model = PeftModel.from_pretrained(model, checkpoint_path)

                model.config.pad_token_id = tokenizer.pad_token_id

            elif adapter == LOFTQ:
                # ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning#load-and-train
                bnb_config.bnb_4bit_use_double_quant = False
                model = AutoModelForSequenceClassification.from_pretrained(
                    model.base_model_path,
                    num_labels=len(OPTION),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                )
                model = PeftModel.from_pretrained(
                    model,
                    base_model_path,
                    subfolder="loftq_init",
                    is_trainable=is_trainable,
                )
                model.config.pad_token_id = tokenizer.pad_token_id
                if is_trainable:
                    model.print_trainable_parameters()

            else:
                raise ValueError(f"Unsupported adapter type: {adapter}")

        elif model_type == BERT:
            model_config = AutoConfig.from_pretrained(base_model_path)
            model = AutoModelForMultipleChoice.from_pretrained(
                base_model_path,
                trust_remote_code=False,
                config=model_config,
            )
            model.to(device)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        raise ValueError(f"Unsupported fine-tune type: {finetune_type}")

    return model
