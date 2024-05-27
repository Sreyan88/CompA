from hook import CLAP_Module
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys

device = torch.device('cuda')

test_dir = sys.argv[1]
class_index_dict_path = sys.argv[2]
ckpt_path = sys.argv[3]

# Load the model
model = CLAP_Module(enable_fusion=True, device=device, amodel= 'HTSAT-tiny', tmodel='t5')
model.load_ckpt(ckpt_path)

# Get the class index dict
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}
all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]

# Get all the data
audio_files = sorted(glob.glob(test_dir + '*.flac', recursive=True))
print(len(audio_files))
json_files = sorted(glob.glob(test_dir + '*.json', recursive=True))
ground_truth_idx = [class_index_dict[json.load(open(jf))['tags'][0]] for jf in json_files]

batch_size = 32

class CustomDataset(Dataset):
    def __init__(self, audio_files, ground_truth):
        self.audio_files = audio_files
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], self.ground_truth[idx]

def custom_collate_fn(batch):
    audio_batch, ground_truth_batch = zip(*batch)
    return audio_batch, ground_truth_batch

custom_dataset = CustomDataset(audio_files, ground_truth_idx)
custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

print(f"Len of loader : {len(custom_loader.dataset)}")

preds = None
for audio_batch, ground_truth_batch in (custom_loader):
    if audio_batch == None:
        break
    with torch.no_grad():
        text_embed = model.get_text_embedding(all_texts)
        audio_embed = model.get_audio_embedding_from_filelist(audio_batch)

        ground_truth = torch.tensor(ground_truth_batch).view(-1, 1)

        ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)
        batch_preds = torch.where(ranking == ground_truth)[1]
        if preds is None:
            preds = batch_preds
        else:
            preds = torch.cat([preds, batch_preds], dim=0)

preds = preds.cpu().numpy()
metrics = {}
metrics[f"mean_rank"] = preds.mean() + 1
metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
for k in [1, 5, 10]:
    metrics[f"R@{k}"] = np.mean(preds < k)
# map@10
metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

print(
    f"Zeroshot Classification Results: "
    + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
)