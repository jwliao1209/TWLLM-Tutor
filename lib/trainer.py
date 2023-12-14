import os

import torch
from constants import CHECKPOINT_DIR
from metric.accuracy import correcter
from metric.perplexity import Perplexity
from tqdm import tqdm
from tracker import MetricTracker

from .utils.train_utils import dict_to_device


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
        self.total_step = 0

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
        # pred_logit = self.model(
        #     input_ids=batch_data["input_ids"],
        #     attention_mask=batch_data["attention_mask"],
        # ).logits
        # ppl = self.eval_func(
        #     pred_logits=pred_logit,
        #     labels=batch_data["input_ids"],
        #     output_masks=batch_data["output_mask"],
        # )
        generated_tokens = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_new_tokens=512,
        )
        generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generations = generations.replace(batch_data["prompt"][0], "").strip()
        correct = int(correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0]))

        # self.tracker.update(f"valid/ppl", ppl, pred_logit.shape[0])
        self.tracker.update(f"valid/acc", correct, 1)
        return

    def log(self, record, step=None):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record, step=step)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]}, step=self.total_step)

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            self.total_step += 1

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

        self.log({"epoch": self.cur_ep, **self.tracker.result()}, step=self.total_step)
        self.progress_bar.close()
        return

    def fit(self, epoch):
        self.model.to(self.device)
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
