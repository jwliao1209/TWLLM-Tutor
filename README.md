# TWLLM Tutor for Secondary Education


## Set the Environment
```
virtualenv --python=python3.10 adl_final
source ~/adl_final/bin/activate
```

```
pip install -r configs/requirements.txt
```


## Prepare Training Dataset
```
PYTHONPATH=lib python prepare_data/convert_train_format.py
```

## Inference
```
PYTHONPATH=lib python infer.py
```

## Reference
- https://www.ceec.edu.tw/files/file_pool/1/0j076464103640279375/04-105%e5%ad%b8%e6%b8%ac%e7%a4%be%e6%9c%83%e7%ad%94%e6%a1%88.pdf

