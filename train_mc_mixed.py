from torch.nn import CrossEntropyLoss
from typing import Optional, Union, Tuple, List
from datasets import Dataset
from datasets.formatting import format_table, get_formatter, query_table
from collections.abc import Sequence
import math
import os
from argparse import ArgumentParser, Namespace
from functools import partial

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, default_data_collator, get_scheduler)

import re
import wandb
from lib.lib_mc.constants import MC_DATA_FILE_WITH_DATABASE
from lib.lib_mc.trainer import MCTrainer
from lib.lib_mc.preprocess import flatten_list, unflatten_list, MC_MAX_SEQ_LEN, MC_ENDING_LEN
from lib.optimizer import get_optimizer
from lib.utils.train_utils import set_random_seeds

from transformers.models.bert import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMultipleChoice, BertModel
from transformers.modeling_outputs import MultipleChoiceModelOutput, BaseModelOutputWithPoolingAndCrossAttentions

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VisibleBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.image2word_proj = torch.nn.Sequential(
            torch.nn.Linear(1024, 768),
            torch.nn.Linear(768, 768),
            torch.nn.Linear(768, 768),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_embeddings = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(
                input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        word_embeddings = self.image2word_proj(image_embeddings)
        print(word_embeddings.shape)
        embedding_output = self.embeddings(
            input_ids=torch.where(input_ids < 0, 0, input_ids),
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class VisibleBertForMultipleChoice(BertForMultipleChoice):
    def __init__(self, config):
        super().__init__(config)
        self.bert = VisibleBertModel(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_embeddings = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)
                                   ) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)
                                         ) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2),
                               inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_embeddings=image_embeddings,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Multiple Choice")

    parser.add_argument("--not_use_mc_train", action="store_true",
                        help="whether not to use data/train_data_mc/train.json")
    parser.add_argument("--not_use_database_mc_train", action="store_true",
                        help="whether not to use data/train_database_mc/train.json")
    parser.add_argument("--valid_data", type=str,
                        default="data/train_data_mc/valid.json",
                        help="valid data")
    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-bert-wwm-ext",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=4,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=10,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="linear",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=300,
                        help="number of warm up steps")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="deivce id")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="Option of train from scratch")
    parser.add_argument("--bf16", action="store_true",
                        help="Option of using bf16")
    parser.add_argument("--only_test", action="store_true",
                        help="Option of do only testing")

    return parser.parse_args()


def replace_and_extract_indices(input_string):
    # Regular expression to find "image{index}" patterns with mixed numbers and alphabets
    pattern = r'image{([a-zA-Z_\d]+)}'

    # Find all matches of the pattern in the input string
    matches = re.finditer(pattern, input_string)

    # Initialize a list to store the extracted indices
    indices = []

    # Replace all matches with "<masked>" and extract indices
    replaced_string = re.sub(pattern, 'photo', input_string)
    for match in matches:
        indices.append(match.group(1))  # Capture the index as a string

    return replaced_string, indices


def preprocess_mc_func(data: dict, tokenizer: AutoTokenizer, train=True) -> dict:
    """
    Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py
    """
    first_sentences = [[context] * 4 for context in data['question']]
    years = [[context] * 4 for context in data['year']]
    second_sentences = [
        [data['A'][i], data['B'][i], data['C'][i], data['D'][i]]
        for i in range(len(data['A']))
    ]

    first_sentences = flatten_list(first_sentences)
    second_sentences = flatten_list(second_sentences)
    years = flatten_list(years)

    temp = []
    for sentence in first_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices))
    first_sentences = [d[0] for d in temp]
    first_indices = [d[1] for d in temp]

    temp = []
    for sentence in second_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices))
    second_sentences = [d[0] for d in temp]
    second_indices = [d[1] for d in temp]

    merged_indices = [
        first_indices[i] + second_indices[i]
        for i in range(len(second_indices))
    ]

    tokenized_data = tokenizer(
        first_sentences,
        second_sentences,
        max_length=MC_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )

    MAX_NUM_EMBS = 10
    image_embeddings = []
    for i, raw_indices in enumerate(tokenized_data['input_ids']):
        embeddings = torch.zeros(MAX_NUM_EMBS, 1024)
        if len(merged_indices[i]) == 0:
            image_embeddings.append(embeddings)
            continue

        image_count = 0
        feature_mapping = {}
        for j in range(len(raw_indices)):
            if raw_indices[j] == 9020:  # 9020 == "photo"
                photo_name = merged_indices[i][image_count]
                if photo_name not in feature_mapping:
                    current_size = len(feature_mapping)
                    feature_mapping[photo_name] = -1 - current_size

                    emb_filename = f"./data/train_data_mc/embeddings/{years[i]}_{photo_name}.pth"
                    embeddings[current_size] = torch.load(emb_filename)

                raw_indices[j] = feature_mapping[photo_name]
                image_count += 1

        image_embeddings.append(embeddings)
        # Use <= not == because there's a truncated case.
        assert image_count <= len(merged_indices[i]), f"{image_count} != {len(merged_indices[i])}" \
            f"{first_sentences[i]} {second_sentences[i]}"

    tokenized_data["image_embeddings"] = image_embeddings
    tokenized_data = {k: unflatten_list(
        v, MC_ENDING_LEN) for k, v in tokenized_data.items()}

    if train:
        tokenized_data["labels"] = data['answer']

    return tokenized_data


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )

    datasets = load_dataset("json", data_files={
        'train_data_mc': "data/train_data_mc/train.json",
        'train_database_mc': "data/train_database_mc/train.json",
        'valid': args.valid_data,
    })

    preprocess_func = partial(preprocess_mc_func, tokenizer=tokenizer)
    processed_datasets = datasets.map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["train_database_mc"].column_names
    )

    if args.not_use_mc_train:
        processed_datasets["train"] = processed_datasets["train_database_mc"]
    elif args.not_use_database_mc_train:
        processed_datasets["train"] = processed_datasets["train_data_mc"]
    else:
        processed_datasets["train"] = concatenate_datasets([
            processed_datasets["train_data_mc"],
            processed_datasets["train_database_mc"]
        ])

    # processed_datasets["valid"] = concatenate_datasets([processed_datasets["valid"], processed_datasets["test"]])
    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        processed_datasets["valid"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.train_from_scratch:
        model = AutoModelForMultipleChoice.from_config(model_config).to(device)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=False,
            config=model_config,
        )
        model: BertForMultipleChoice = model
        state_dict = model.state_dict()
        model = VisibleBertForMultipleChoice(model.config).to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            print("Some keys are missing")
            model.load_state_dict(state_dict, strict=False)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="adl-final",
        group="mc",
        name="experiment_mc",
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_warmup_steps": args.warm_up_step,
        }
    )
    wandb.watch(model, log="all")

    trainer = MCTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
        bf16=args.bf16,
    )

    if args.only_test:
        trainer.valid_one_epoch()
    else:
        trainer.fit(epoch=args.epoch)
    wandb.finish()
