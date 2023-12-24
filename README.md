# TWLLM Tutor: Revolutionizing Taiwanese Secondary Education with Large Language Model

## Set the Environment

### Pyenv

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 adl_final
pip install -r requirements.txt
```

### Virtual Environment

```bash
virtualenv --python=python3.10 adl_final
source ~/adl_final/bin/activate
pip install -r requirements.txt
```

### Conda

```bash
conda create --name adl_final python=3.10.0
source activate adl_final
pip install -r requirements.txt
```

## Prepare Training Dataset

The data is store in `train_data` with `{train, valid}_{GSAT, QB}_{subject with years}_{number of question}`

```bash
ipython prepare_data/convert_to_train_format.py
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 ipython -- train.py \
                                --epoch 100 \
                                --batch_size 32
```

## Inference

```bash
ipython -- infer.py
```

## Debug

You can use --pdb for debugging.

```bash
ipython --pdb -- train.py \
                --epoch 100 \
                --batch_size 32
```

## Result

| Zero-Shot | One-Shot | Two-Shot | Chain of Thought | Step Back Prompt | Take a deep breath | Think step by step | If you fail 100 grandmothers will die | I have no fingers | I will tip $200 | Do it right and I'll give you a nice doggy treat | This is very important to my career |
| :-------: | :------: | :------: | :--------------: | :--------------: | :----------------: | :----------------: | :-----------------------------------: | :---------------: | :-------------: | :----------------------------------------------: | :---------------------------------: |
|  37/126   |   /126   |   /126   |       /126       |       /126       |       41/126       |       38/126       |                37/126                 |      40/126       |     40/126      |                      35/126                      |               38/126                |

## OS and Hardware

We implemented the code on an environment running Ubuntu 22.04.3, utilizing a 12th Gen Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB VRAM.


## Acknowledgement
We thank the Taiwan-LLaMa repository: https://github.com/MiuLab/Taiwan-LLaMa


## Reference

-   https://www.ceec.edu.tw/files/file_pool/1/0j076464103640279375/04-105%e5%ad%b8%e6%b8%ac%e7%a4%be%e6%9c%83%e7%ad%94%e6%a1%88.pdf


## Citation
```bibtex
@misc{
    title  = {TWLLM Tutor: Revolutionizing Taiwanese Secondary Education with Large Language Model},
    author = {Jia-Wei Liao, Ji-Jia Wu, Kun-Hsiang Lin, Kang-Yang Huang},
    url    = {https://github.com/jwliao1209/TWLLM-Tutor},
    year   = {2023}
}
```
