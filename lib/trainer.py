import os
import torch
from tqdm import tqdm

from constants import CHECKPOINT_DIR
from tracker import MetricTracker
from metric.perplexity import Perplexity
from metric.accuracy import get_correct_num, correcter
from utils.train_utils import dict_to_device
from utils.data_utils import write_json


class Trainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        *arg,
        **kwarg,
        ):

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        self.eval_func = Perplexity()
        self.tracker = MetricTracker()
        self.logger = logger

    def train_step(self, batch_data, index):
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        n = preds.shape[0]
        self.tracker.update("train/loss", loss / n, n)
        return loss

    def valid_step(self, batch_data, index):
        generated_tokens = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_new_tokens=512,
        )
        generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generations = generations.replace(batch_data["prompt"][0], "").strip()
        is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])

        self.tracker.update(f"valid/acc", int(is_correct), 1)
        return {
            "id": int(batch_data['id'][0]),
            "year": batch_data['year'][0],
            "prompt": batch_data['prompt'][0],
            "generation": generations,
            "answer": batch_data['answer'][0],
            "answer_details": batch_data['answer_description'][0],
            "is_correct": is_correct,
        } 

    def log(self, record):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        self.progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/acc"])

        pred_list = []
        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            self.progress_bar.set_postfix(self.tracker.result())
            pred = self.valid_step(batch_data, step)
            pred_list.append(pred)

        self.log({"epoch": self.cur_ep, **self.tracker.result()})
        self.progress_bar.close()

        os.makedirs("predictions", exist_ok=True)
        write_json(pred_list, os.path.join("predictions", f"lora_{self.cur_ep}.json"))
        return

    def fit(self, epoch):
        self.model.to(self.device)
        self.cur_ep = 0
        self.valid_one_epoch()
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
            self.model.save_pretrained(
                os.path.join(
                CHECKPOINT_DIR,
                f"epoch={self.cur_ep}_acc={self.tracker.result().get('valid/acc', 0)}"
            )
        )
        return


class MCTrainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        *arg,
        **kwarg,
        ):

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        self.eval_func = Perplexity()
        self.tracker = MetricTracker()
        self.logger = logger

    def train_step(self, batch_data, index):
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        n = preds.shape[0]
        self.tracker.update("train/loss", loss / n, n)
        return loss

    def valid_step(self, batch_data, index):
        preds = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
        ).logits.argmax(dim=-1)
        correct_num = get_correct_num(preds, batch_data["labels"])
        self.tracker.update(f"valid/acc", correct_num, preds.shape[0])
        return

    def log(self, record):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        self.progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/acc"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            self.progress_bar.set_postfix(self.tracker.result())
            self.valid_step(batch_data, step)

        self.log({"epoch": self.cur_ep, **self.tracker.result()})
        self.progress_bar.close()
        return

    def fit(self, epoch):
        self.model.to(self.device)
        self.cur_ep = 0
        self.valid_one_epoch()
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
            self.model.save_pretrained(
                os.path.join(
                CHECKPOINT_DIR,
                f"epoch={self.cur_ep}_acc={self.tracker.result().get('valid/acc', 0)}"
            )
        )
        return
