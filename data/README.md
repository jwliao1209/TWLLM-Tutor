# Data

## Directory Structure

```
data
|- public       # Public is for data we consider publishable (without copyright issues, etc).
|- private      # Private is for data we don't want to publish (For future extensions).
|- raw_data     # Raw_data is a directory to save raw data.
```

## Data Format

We use a directory to save Each textual data.
```
{data_name}
|- metadata.txt # The metadata of this data
|- content.json # The main content of the data
|- *.png        # The raw data of each figure in the data
|- *.pth        # The image embedding of each figure in the data
|- *.md         # The text in markdown for each table in the data
```

### Content

For each year's [university exam](https://www.ceec.edu.tw/xmfile?xsmsid=0J052424829869345634), we parse the question and answer as the following format
```json
{
    "question_groups": [
        {
            "ids": [12,13,14],
            "prefix": "某財經專業雜誌，刊出下列新聞：..."
        },
        {
            "ids": [64,65,66],
            "prefix": "\\image{9}是1848年至1849年歐洲一系列武裝革命運動的分布情形，這波革命運動雖然都以失敗告終，但對歐洲的影響卻極為深遠。請問："
        }
    ]
    "questions": [
        {
            "id": 2,
            "type": "single",
            "answer": "C",
            "question": "時下各國普遍流行將國營事業開放民間經營，試問這種決策的主要著眼點為何？",
            "A": "實現社會公平",
            "B": "增進社會福利",
            "C": "提高經營效率",
            "D": "揚棄共產主義",
        }, 
        {
            "id": 72,
            "type": "multi",
            "answer": "AD",
            "question": "邊疆民族進入塞內，占有原為漢族之土地而統冶之，若以不同制度冶理本族與漢族，即稱「二元政冶」，在中國歷史上，採「二元政治」的朝代有那些？",
            "A": "遼",
            "B": "西夏",
            "C": "金",
            "D": "元",
            "E": "清",
        },
        {
            "id": 13,
            "type": "single",
            "answer": "B",
            "question": "該立法主張如獲得通過，最可能侵害以下哪一種人民權益？",
            "A": "公司契約自由",
            "B": "媒體新聞自由",
            "C": "民眾媒體近用",
            "D": "記者個人姓名",
        }
    ]
}
```


