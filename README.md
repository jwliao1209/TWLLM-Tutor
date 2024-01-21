# Taiwan-LLM Tutor: Large Language Models for Taiwanese Secondary Education

<img width="800" alt="twllm" src="https://github.com/jwliao1209/Taiwan-LLM-Tutor/assets/55970911/a9d5895b-dd95-4609-95ec-69a01bca238b">


## Setup

### Conda

```
conda create --name twllm_tutor python=3.10.0
source activate twllm_tutor
pip install -r requirements.txt
```

### Virtual Environment

```
virtualenv --python=python3.10.0 twllm_tutor
source ~/twllm_tutor/bin/activate
pip install -r requirements.txt
```

### Pyenv

```
pyenv install 3.10.0
pyenv virtualenv 3.10.0 twllm_tutor
pip install -r requirements.txt
```


## Dataset

### GSAT Dataset

The GSAT social dataset is downloaded from [GSAT Website](https://www.ceec.edu.tw/files/file_pool/1/0j076464103640279375/04-105%e5%ad%b8%e6%b8%ac%e7%a4%be%e6%9c%83%e7%ad%94%e6%a1%88.pdf).

| Dataset | Year | Number of Question | 
|:-|:-|:-|
| train_GSAT_social  | 83-107  | 1221 |
| valid_GSAT_history | 108-112 | 97   |
| valid_GSAT_civics  | 108-112 | 79   |
| valid_GSAT_social  | 108-112 | 213  |

### Question Bank Dataset

| Dataset | Number of Question |
|:-|:-|
| train_QB_geography | 312  |
| train_QB_history   | 9000  |
| train_QB_civics    | 2035  |
| train_QB_social    | 11347 |
| valid_QB_history   | 205   |


### File Structure

```
./data
   |─ raw_data
   |   |─ GSAT_social  # Public is for data we consider publishable (without copyright issues, etc).
   |   └─ QB_social    # Private is for data we don't want to publish (For future extensions).
   └─ train_data
       |- GSAT_social
       |- GSAT_social_with_image
       └─ QB_social
```

<!---
The data stored in `data/train_data/GSAT_social_with_image` has been preprocessed using the following commands:

```bash
python prepare_data/convert_vision_mc_format.py
python prepare_data/prepare_embeddings.py
```
--->

### Data Format

```
{
        "subject": "social_study",
        "year": "83",
        "id": 1,
        "type": "single",
        "question": "孫中山先生認為造成中國人像一盤散沙，民族不夠團結的主因為何",
        "A": "任外族帝制專斷的統治下，人民喪失了關心公共事務的能力",
        "B": "異族的征服者過於強大，中國人團結也沒用",
        "C": "中國入的家族觀念過於發達",
        "D": "過早提倡天下一家的世界主義",
        "answer": "A",
        "answer_details": ""
},
```


## Config

To generate the config, you can run the command:
```
python generate_config.py
```


## Training

### Fine-Tune Chinese-BERT

- Chinese-BERT for Multiple Choice
- Vision-BERT for Multiple Choice

To fine-tune the Chinese-BERT, you can run the command:
```bash
bash scripts/run_train_bert.sh
```

### Fine-Tune Taiwan-LLM

#### QLoRA Adapter

- Instruction Tuning

```bash
FINETUNE_METHOD=qlora_instruction_tuning bash scripts/run_train_twllm.sh
```

- Multiple Choice

```bash
FINETUNE_METHOD=qlora_multiple_choice bash scripts/run_train_twllm.sh
```

#### LoftQ Adapter

- Quantize Model for LoftQ

```bash
bash scripts/run_quantize_twllm_loftq.sh
```

- Instruction Tuning

```bash
FINETUNE_METHOD=loftq_instruction_tuning bash scripts/run_train_twllm.sh
```

- Multiple Choice
```bash
FINETUNE_METHOD=loftq_multiple_choice bash scripts/run_train_twllm.sh
```

## Testing

```bash
python test.py
```


## GSAT Question and Answer Example
| Question  | **Taiwan-LLM Tutor** | ChatGPT | 
|:-|:-|:-|
| 你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。十九世紀前期，張三從寧波來艋舺採購樟腦，計畫運回浙江、上海銷售，同時也運來綢緞、布帛等貨品來臺銷售。請問：張三最可能委託下列哪個組織運送這批貨物？ A.東印度公司 B.北郊 C.怡和洋行 D.三井會社 | B.北郊 原因：當時臺北港的北郊商人常兼職貿易商，對於貨物的運送和買賣多半有涉獵。✅ | C. 怡和洋行 原因：在十九世紀前期，張三要將樟腦、綢緞、布帛等貨品運送回浙江和上海銷售，最可能委託的組織是怡和洋行。怡和洋行是一家在19世紀初期在華南地區經營貿易的外國商行，主要從中國向其他國家運送貨物。因此，怡和洋行是最可能協助張三運送這批貨物的組織。其他選項，如A. 東印度公司、B. 北郊、D. 三井會社，都不太可能參與這個情境中的貨物運送。 | 
| 你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。學者指出：北魏實行均田法的背景，第一，因戰亂的傷亡及遷徙，留下大片無主荒田；第二，許多農民為躲避戰禍，逃離農村，流落城市，成為不事生產的游民；第三，農民投靠大戶士族，「五十三十家方為一戶」，只向其宗主貢獻，不必向政府納稅。均田法的實施是結合戶籍的清查及管理，透過計口授田，使農業勞動力與耕地達到合理配合，以恢復生產，建立小農社會。根據上述，學者認為北魏均田法實施的主要目的最可能是：A.限制土地兼併 B.保全士族特權 C.重建編戶稅基 D.推行漢化運動 | C.重建編戶稅基 原因：北魏均田法的實施，使農民戶籍清查並管理，透過計口授田，使農業勞動力與耕地達到合理配合，以恢復生產，重建編戶稅基。\n(A)與資料無關。\n(B)與資料無關。\n(D)與資料無關。✅ | A. 限制土地兼併 原因：根據提供的資訊，北魏實行均田法的背景包括大片無主荒田、許多農民流落城市成為游民、農民投靠大戶士族等情況。均田法的實施主要目的是限制土地兼併，通過計口授田，確保農業勞動力和土地之間的合理配合，防止土地被少數大戶士族壟斷，促進農業生產，建立小農社會。因此，A. 限制土地兼併是最合適的答案。 | 
| 你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。1949年10月，共軍集中船隻對金門發動攻擊，遭守軍擊退。1950年3月，共軍又集結大批兵力，用類似「萬船齊發戰術」進攻海南島，迫使當地國軍撤守；5月底更攻占廣東萬山群島。共軍原本預定隨後再度對金門發動攻擊，但沒有多久就取消這項計畫。最可能的原因是：A.共軍軍備及船隻有限，已經無法再負擔長期作戰 B.共軍顧忌金門兵力強大，寧可保持戰果故而避戰 C.韓戰爆發，美國宣布派遣第七艦隊協助臺灣防務 D.臺灣實施戒嚴，共軍乃轉以心戰策略來瓦解國軍。 | C.韓戰爆發，美國宣布派遣第七艦隊協助臺灣防務 原因：韓戰爆發後，美國宣布派遣第七艦隊協助臺灣防務，因此共軍暫緩對金門的攻擊。✅ | A. 共軍軍備及船隻有限，已經無法再負擔長期作戰 原因：1949年10月，共軍對金門發動攻擊，但遭到守軍擊退。隨後，共軍又進攻海南島並攻占廣東萬山群島。然而，最可能的原因取消對金門的再次攻擊計畫是因為共軍的軍備和船隻有限，已經難以再負擔長期作戰，尤其在面對金門守軍強大的情況下。共軍可能決定將資源用於其他地區或其他策略，因為攻擊金門可能會導致更大的損失。其他選項如B. 共軍顧忌金門兵力強大、C. 韓戰爆發，... | 


## OS and Hardware

The experiments were performed on a personal computer equipped with a single NVIDIA GeForce RTX 4090 GPU with 24 GB of VRAM, and a server configuration featuring a single RTX A6000 GPU with 49 GB of VRAM.


## Acknowledgements

- [Taiwan-LLM](https://github.com/MiuLab/Taiwan-LLM)
- [transformers](https://github.com/huggingface/transformers)
- [LoftQ](https://github.com/yxli2123/LoftQ)
- [lion-pytorch repository](https://github.com/lucidrains/lion-pytorch)


## Citation

```bibtex
@misc{liao2023twllm
    title  = {Taiwan-LLM Tutor: Large Language Models for Taiwanese Secondary Education},
    author = {Jia-Wei Liao, Ji-Jia Wu, Kun-Hsiang Lin, Kang-Yang Huang},
    url    = {https://github.com/jwliao1209/TWLLM-Tutor},
    year   = {2023}
}
```
