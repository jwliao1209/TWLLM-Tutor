import os
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .constants import (CHECKPOINT_DIR, LABEL_TO_OPTION, MAX_NEW_TOKENS,
                        PREDICTION_DIR)
from .metric.accuracy import correcter, get_correct_num
from .metric.perplexity import Perplexity
from .tracker import MetricTracker
from .utils.data_utils import write_json
from .utils.train_utils import dict_to_device


class BaseTrainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        optimizer=None,
        lr_scheduler=None,
        accum_grad_step: int = 1,
        clip_grad_norm: Optional[float] = 1.0,
        fp32: bool = False,
        logger=None,
        disable_valid_on_start: bool = False,
        checkpoint_dir: Optional[str] = None,
        prediction_dir: Optional[str] = None,
        max_new_token: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_num = len(train_loader) if train_loader is not None else 0
        self.valid_num = len(valid_loader) if valid_loader is not None else 0
        self.test_num = len(test_loader) if test_loader is not None else 0
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler = lr_scheduler
        self.eval_func = Perplexity()
        self.tracker = MetricTracker()
        self.fp32 = fp32
        self.grad_scaler = GradScaler(enabled=not fp32)
        self.logger = logger
        self.disable_valid_on_start = disable_valid_on_start
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
        self.prediction_dir = prediction_dir if prediction_dir is not None else PREDICTION_DIR
        self.max_new_token = max_new_token if max_new_token is not None else MAX_NEW_TOKENS
        print(self)

    def train_step(self, batch_data, index):
        NotImplementedError

    def valid_step(self, batch_data, index):
        NotImplementedError

    def log(self, record):
        if self.logger is not None:
            self.logger.log(record)
        return

    # TODO: refactor
    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" and not self.fp32 else torch.float32
            ):
                loss = self.train_step(batch_data, step)

            self.progress_bar.set_postfix({"train_loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({"train/loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})

            # ref: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
            self.grad_scaler.scale(loss / self.accum_grad_step).backward()

            if step % self.accum_grad_step == 0:
                self.grad_scaler.unscale_(self.optimizer)
                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.grad_scaler.step(optimizer=self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()

    # TODO: refactor
    @torch.no_grad()
    def valid_one_epoch(self, valid_loader=None):
        self.model.eval()
        valid_loader = self.valid_loader if valid_loader is None else valid_loader
        self.progress_bar = tqdm(valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/acc"])

        pred_list = []
        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            self.progress_bar.set_postfix(self.tracker.result())

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" and not self.fp32 else torch.float32
            ):
                pred = self.valid_step(batch_data, step)
            pred_list.append(pred)

        self.log({"epoch": self.cur_ep} | self.tracker.result())
        self.progress_bar.close()

        save_name = f"epoch={self.cur_ep}_acc={self.tracker.result().get('valid/acc', 0):.4f}"
        os.makedirs(self.prediction_dir, exist_ok=True)
        write_json(pred_list, os.path.join(self.prediction_dir, f"{save_name}.json"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(os.path.join(self.checkpoint_dir, save_name))

    def fit(self, epoch):
        self.cur_ep = 0
        if not self.disable_valid_on_start:
            self.valid_one_epoch()
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()

    def __repr__(self):
        tab = " " * 2
        x = (
            f"{self.__class__.__name__}(\n"
            f" tokenizer={self.tokenizer},\n"
            f" model={self.model},\n"
            f" device={self.device},\n"
            f" train_loader={self.train_loader},\n"
            f" valid_loader={self.valid_loader},\n"
            f" train_num={self.train_num},\n"
            f" valid_num={self.valid_num},\n"
            f" optimizer={self.optimizer},\n"
            f" train_batch_size={self.train_loader.batch_size if self.train_loader is not None else None},\n"
            f" accum_grad_step={self.accum_grad_step},\n"
            f" clip_grad_norm={self.clip_grad_norm},\n"
            f" lr_scheduler={self.lr_scheduler},\n"
            f" disable_valid_on_start={self.disable_valid_on_start},\n"
            f" checkpoint_dir={self.checkpoint_dir},\n"
            f" prediction_dir={self.prediction_dir},\n"
            f" max_new_token={self.max_new_token}"
        )
        x = x.replace("\n", f"\n{tab}").replace(f'\n{tab}P', f'\n{tab}{tab}P') + "\n)"
        return x


class InstructionTuningTrainer(BaseTrainer):

    def train_step(self, batch_data, index):
        b = batch_data["input_ids"].shape[0]
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        )
        loss = outputs.loss
        return loss

    def valid_step(self, batch_data, index):
        generated_tokens = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_new_tokens=self.max_new_token,
        )

        generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generations = generations.replace(batch_data["prompt"][0], "").strip()
        is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])

        self.tracker.update(f"valid/acc", int(is_correct), 1)
        output_dict = dict(
            id=int(batch_data['id'][0]),
            year=batch_data['year'][0],
            prompt=batch_data['prompt'][0],
            generation=generations,
            answer=batch_data['answer'][0],
            answer_details=batch_data['answer_description'][0],
            is_correct=is_correct,
        )
        return output_dict

    @torch.no_grad()
    def test(self, test_loader=None):
        test_loader = self.test_loader if test_loader is None else test_loader
        correct_num = 0
        prediction_list = []
        test_bar = tqdm(test_loader, desc=f"Testing")
        for _, batch_data in enumerate(test_bar, start=1):
            with torch.no_grad():
                batch_data = dict_to_device(batch_data, self.device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    generated_tokens = self.model.generate(
                        input_ids=batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                        max_new_tokens=self.max_new_token,
                    )
                generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                generations = generations.replace(batch_data["prompt"][0], "").strip()

                is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])
                if is_correct:
                    correct_num += 1

                test_bar.set_postfix({"correct_num": correct_num})

                print(f"Question:\n{batch_data['prompt'][0]}\n")
                print(f"Answer:\n{batch_data['answer'][0]}\n")
                print(f"Prediction:\n{generations}\n")
                print(f"Is Correct: {is_correct}")

                prediction_list.append(
                    {
                        "id": int(batch_data['id'][0]),
                        "year": batch_data['year'][0],
                        "prompt": batch_data['prompt'][0],
                        "generation": generations,
                        "answer": batch_data['answer'][0],
                        "answer_details": batch_data['answer_description'][0],
                        "is_correct": is_correct,
                    }
                )

        print('Acc:', correct_num / len(test_loader))
        return prediction_list


class MultipleChoiceTrainer(BaseTrainer):

    def train_step(self, batch_data, index):
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16 if self.device.type == "cuda" and not self.fp32 else torch.float32
        ):
            outputs = self.model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                labels=batch_data["labels"],
            )
        loss = outputs.loss
        return loss

    def valid_step(self, batch_data, index):
        preds = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
        )
        preds = preds.logits.argmax(dim=-1)
        correct_num = get_correct_num(preds, batch_data["labels"])
        self.tracker.update(f"valid/acc", correct_num, preds.shape[0])
        output_dict = dict(
            id=int(batch_data["id"][0]),
            year=batch_data["year"][0],
            question=batch_data["question"][0],
            generation=LABEL_TO_OPTION[int(preds)],
            answer=batch_data["answer"][0],
            answer_details=batch_data["answer_description"][0],
            is_correct=int(preds == batch_data["labels"]),
        )
        return output_dict
