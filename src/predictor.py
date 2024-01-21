from tqdm import tqdm
from tying import Optional

import torch

from src.constants import CHECKPOINT_DIR, PREDICTION_DIR, MAX_NEW_TOKENS


class Predictor:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        data_loader=None,
        checkpoint_dir: Optional[str] = None,
        prediction_dir: Optional[str] = None,
        max_new_token: int = 128,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.data_num = len(data_loader) if data_loader is not None else 0
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
        self.prediction_dir = prediction_dir if prediction_dir is not None else PREDICTION_DIR
        self.max_new_token = max_new_token
    
    def predict(self):
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
