import clip
from PIL import Image
from glob import glob
import torch.nn.functional as F
import torch

model, preprocess = clip.load("RN50")

for filename in glob('./data/public/university_exams/social_study/*/*.png'):
    img = preprocess(Image.open(filename))
    year, id = filename.split("/")[-2:]
    id = id[:-4]
    output_filename = f"./data/train_data_mc/embeddings/{year}_{id}.pth"

    emb = model.encode_image(img[None].cuda())
    emb = F.normalize(emb, p=2, dim=-1).cpu()
    torch.save(emb, output_filename)
    print(output_filename)


