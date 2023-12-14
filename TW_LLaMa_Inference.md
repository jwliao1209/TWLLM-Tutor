## Inference
### train_data/valid.json (Accuracy)
|Zero-Shot|One-Shot|Two-Shot|Chain of Thought|Step Back Prompt|Take a deep breath|Think step by step|If you fail 100 grandmothers will die|I have no fingers|I will tip $200|Do it right and I'll give you a nice doggy treat|This is very important to my career|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|34/185|/185|/185|/185|/185|36/185|32/185|/185|/185|/185|/185|/185|

```shell
PYTHONPATH=lib python infer.py
```

## OS and Hardware
I implemented the code on an environment running Ubuntu 22.04.3, utilizing a 12th Gen Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB VRAM.