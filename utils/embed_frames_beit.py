# # Prepare submission

import os

import numpy as np
import torch
from transformers import BeitFeatureExtractor, BeitModel
from tqdm import tqdm

from utils.video_dataset import VideoDataset
data_dir = '/data/behavior-representation'

frame_number_map = np.load(os.path.join(data_dir, 'frame_number_map.npy'), allow_pickle=True).item()

video_size = 'full_size'
video_set = 'submission'

video_data_dir = os.path.join(data_dir, 'videos', video_size, video_set)

ds = VideoDataset(video_data_dir, frame_number_map, channels_first=True)

batch_size=32
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=False,
    num_workers=8,
)
print("len(dataloader), ", len(dataloader))

feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-large-patch16-512')
model = BeitModel.from_pretrained('microsoft/beit-large-patch16-512')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

n_samples, embedding_length = len(ds), 1024
embedding_array = np.zeros((n_samples, embedding_length))

cache_path = os.path.join("cache", "beit_embeddings.npz")
os.makedirs("cache", exist_ok=True)

#### This is how to load embeddings ###
# print("Loading embeddings")
# a = np.load(cache_path)
# embedding_array = a["arr_0"]

for i, x in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        inputs = feature_extractor(images=[xx for xx in x], return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = outputs["pooler_output"].cpu().numpy()
        start_idx, end_idx = i * batch_size, (i+1) * batch_size
        embedding_array[start_idx:end_idx] = embeddings
        if i % 1000 == 0:
            print(f"Caching embedding array to {cache_path}")
            np.savez(cache_path, embedding_array)

print(f"Caching embedding array to {cache_path}")
np.savez(cache_path, embedding_array)
