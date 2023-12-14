import os

import torch
from tqdm import tqdm

from ..metric.accuracy import get_correct_num
from ..utils.train_utils import dict_to_device
from .constants import CHECKPOINT_DIR
from .tracker import MetricTracker


class BaseTrainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        bf16=False,
        *arg,
        **kwarg,
    ):

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        self.tracker = MetricTracker()
        self.logger = logger
        self.bf16 = bf16
        self.cur_ep = 0
        self.currunt_step = 0

    def train_step(self, batch_data, step):
        NotImplementedError

    def valid_step(self, batch_data, step):
        NotImplementedError

    def log(self, record, step=None):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record, step=step)
        return

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}", unit_scale=self.train_loader.batch_size)
        self.tracker.reset(keys=["train/acc"])
        acc_loss = 0
        for step, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            with torch.cuda.amp.autocast(enabled=self.bf16, dtype=torch.bfloat16 if self.bf16 else torch.float32):
                loss = self.train_step(batch_data, step)

            loss = (loss / self.accum_grad_step)
            loss.backward()
            acc_loss += loss
            if step % self.accum_grad_step == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.log({"train/loss": acc_loss.item(), "lr": lr}, step=self.currunt_step)
                progress_bar.set_postfix({'loss': acc_loss.item(), "lr": lr})

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                acc_loss = 0

            self.currunt_step += 1

        progress_bar.set_postfix({'loss': acc_loss.item(), "lr": lr, **self.tracker.result()})
        self.log(self.tracker.result(), step=self.currunt_step)

        progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        self.tracker.reset(keys=["valid/loss", "valid/acc"])

        progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        for step, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            self.valid_step(batch_data, step)
            progress_bar.set_postfix(self.tracker.result())

        self.log({**self.tracker.result()}, step=self.currunt_step)
        progress_bar.close()
        self.model.save_pretrained(
            os.path.join(
                CHECKPOINT_DIR,
                f"mc_epoch={self.cur_ep}_acc={self.tracker.result().get('valid/acc', 0):.4f}"
            )
        )
        return

    def fit(self, epoch):
        self.model.to(self.device)
        self.valid_one_epoch()
        for _ in range(epoch):
            self.cur_ep += 1
            self.train_one_epoch()
            self.valid_one_epoch()
        return


class MCTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        bf16=False,
        *arg,
        **kwarg,
    ):
        super().__init__(
            model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            accum_grad_step,
            lr_scheduler,
            logger,
            bf16,
        )

    def share_step(self, batch_data, index, prefix):
        outputs = self.model(**batch_data)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        correct_num = get_correct_num(preds, batch_data["labels"])
        n = preds.shape[0]

        self.tracker.update(f"{prefix}/acc", correct_num / n, n)

        return loss

    def train_step(self, batch_data, index):
        return self.share_step(batch_data, index, "train")

    def valid_step(self, batch_data, index):
        return self.share_step(batch_data, index, "valid")
