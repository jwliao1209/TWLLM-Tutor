import os
from typing import Optional
from collections import OrderedDict

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .constants import CHECKPOINT_DIR
from .constants import PREDICTION_FILE, CONFIG_FILE
from .constants import LABEL_TO_OPTION
from .metric.accuracy import correcter, get_correct_num
from .tracker import MetricTracker
from .utils.data_utils import write_json, save_config
from .utils.train_utils import dict_to_device


class BaseTrainer:
    """
    A class of base trainer.
    """

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
        logger=None,
        accum_grad_step: int = 1,
        clip_grad_norm: Optional[float] = 1.0,
        fp32: bool = False,
        disable_valid_on_start: bool = False,
        max_new_token: int = 128,
        checkpoint_dir: Optional[str] = None,
        config: Optional[dict] = None,
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
        self.tracker = MetricTracker()
        self.fp32 = fp32
        self.grad_scaler = GradScaler(enabled=not fp32)
        self.logger = logger
        self.disable_valid_on_start = disable_valid_on_start
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
        self.max_new_token = max_new_token
        self.config = config
        self.cur_ep = 0
        print(self)

    def train_step(self, batch_data):
        raise NotImplementedError

    def valid_step(self, batch_data):
        raise NotImplementedError

    def log(self, record):
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.model.train()

        for step, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" \
                    and not self.fp32 else torch.float32
            ):
                loss = self.train_step(batch_data)

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

            record = {
                "train_loss": loss.item(),
                "lr": self.lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(record)
            self.log(record)

        progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/acc"])
        self.model.eval()

        pred_list = []
        for _, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            progress_bar.set_postfix(self.tracker.result())

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" \
                    and not self.fp32 else torch.float32
            ):
                pred = self.valid_step(batch_data)
            pred_list.append(pred)

        self.log({"epoch": self.cur_ep} | self.tracker.result())

        save_folder = os.path.join(
            self.checkpoint_dir,
            f"epoch={self.cur_ep}_acc={self.tracker.result().get('valid/acc', 0):.4f}"
        )

        # save prediction
        write_json(pred_list, os.path.join(save_folder, PREDICTION_FILE))

        # save model weight
        self.model.save_pretrained(save_folder)

        # save config
        if self.config is not None:
            save_config(self.config, os.path.join(save_folder, CONFIG_FILE))

        progress_bar.close()
        return

    def fit(self, epoch):
        if not self.disable_valid_on_start:
            self.valid_one_epoch()
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
        return

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
            f" train_batch_size={self.train_loader.batch_size},\n"
            f" accum_grad_step={self.accum_grad_step},\n"
            f" clip_grad_norm={self.clip_grad_norm},\n"
            f" lr_scheduler={self.lr_scheduler},\n"
            f" disable_valid_on_start={self.disable_valid_on_start},\n"
            f" checkpoint_dir={self.checkpoint_dir},\n"
            f" max_new_token={self.max_new_token}"
        )
        return x.replace("\n", f"\n{tab}").replace(f'\n{tab}P', f'\n{tab}{tab}P') + "\n)"


class InstructionTuningTrainer(BaseTrainer):
    """
    A class of trainer for instruction tuning.
    """

    def train_step(self, batch_data):
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
            use_cache=False,
        )
        return outputs.loss

    def valid_step(self, batch_data):
        generated_tokens = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_new_tokens=self.max_new_token,
        )

        generations = self.postprocess(
            predition=generated_tokens,
            prompt=batch_data["prompt"][0],
        )

        is_correct = correcter(
            generation=generations,
            answer=batch_data['answer'][0],
            description=batch_data['answer_description'][0]
        )

        self.tracker.update("valid/acc", int(is_correct), 1)

        output_dict = OrderedDict(
            year=batch_data["year"][0],
            id=int(batch_data["id"][0]),
            prompt=batch_data["prompt"][0],
            generation=generations,
            answer=batch_data["answer"][0],
            answer_details=batch_data["answer_description"][0],
            is_correct=is_correct,
        )
        return output_dict

    def postprocess(self, predition, prompt):
        generation = self.tokenizer.batch_decode(predition, skip_special_tokens=True)[0]
        return generation.replace(prompt, "").strip()

    # @torch.no_grad()
    # def test(self, test_loader=None):
    #     test_loader = self.test_loader if test_loader is None else test_loader
    #     correct_num = 0
    #     prediction_list = []
    #     test_bar = tqdm(test_loader, desc="Testing")
    #     for _, batch_data in enumerate(test_bar, start=1):
    #         with torch.no_grad():
    #             batch_data = dict_to_device(batch_data, self.device)
    #             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #                 generated_tokens = self.model.generate(
    #                     input_ids=batch_data["input_ids"],
    #                     attention_mask=batch_data["attention_mask"],
    #                     max_new_tokens=self.max_new_token,
    #                 )
    #             generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    #             generations = generations.replace(batch_data["prompt"][0], "").strip()

    #             is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])
    #             if is_correct:
    #                 correct_num += 1

    #             test_bar.set_postfix({"correct_num": correct_num})

    #             print(f"Question:\n{batch_data['prompt'][0]}\n")
    #             print(f"Answer:\n{batch_data['answer'][0]}\n")
    #             print(f"Prediction:\n{generations}\n")
    #             print(f"Is Correct: {is_correct}")

    #             prediction_list.append(
    #                 {
    #                     "year": batch_data['year'][0],
    #                     "id": int(batch_data['id'][0]),
    #                     "prompt": batch_data['prompt'][0],
    #                     "generation": generations,
    #                     "answer": batch_data['answer'][0],
    #                     "answer_details": batch_data['answer_description'][0],
    #                     "is_correct": is_correct,
    #                 }
    #             )

    #     print('Acc:', correct_num / len(test_loader))
    #     return prediction_list


class MultipleChoiceTrainer(BaseTrainer):
    """
    A class of trainer for multiple choice.
    """

    def train_step(self, batch_data):
        return self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        ).loss

    def valid_step(self, batch_data):
        preds = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
        ).logits.argmax(dim=-1)

        correct_num = get_correct_num(preds, batch_data["labels"])
        self.tracker.update("valid/acc", correct_num, preds.shape[0])

        output_dict = OrderedDict(
            id=int(batch_data["id"][0]),
            year=batch_data["year"][0],
            question=batch_data["question"][0],
            generation=LABEL_TO_OPTION[int(preds)],
            answer=batch_data["answer"][0],
            answer_details=batch_data["answer_description"][0],
            is_correct=int(preds == batch_data["labels"]),
        )
        return output_dict
