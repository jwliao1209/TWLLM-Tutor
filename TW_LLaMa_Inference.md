## Inference
### train_data/valid.json (Accuracy)
|Zero-Shot|One-Shot|Two-Shot|Step Back Prompt|Take a deep breath|Let's think step by step (Zero-Shot CoT)|If you fail 100 grandmothers will die|I have no fingers|I will tip $200|Do it right and I'll give you a nice doggy treat|This is very important to my career|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|41/126|/126|/126|/126|35/126|38/126|37/126|40/126|40/126|35/126|38/126|

```shell
PYTHONPATH=lib python infer.py
```

## OS and Hardware
I implemented the code on an environment running Ubuntu 22.04.3, utilizing a 12th Gen Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB VRAM.
