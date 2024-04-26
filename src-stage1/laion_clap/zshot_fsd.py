from hook import CLAP_Module
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda')

# download https://drive.google.com/drive/folders/1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX?usp=sharing and extract ./ESC50_1/test/0.tar to ./ESC50_1/test/
esc50_test_dir = '.CLAP/FSD50K.eval_audio/processed/test/'
class_index_dict_path = '.CLAP/class_labels/FSD50k_class_labels_indices.json'

# Load the model
model = CLAP_Module(enable_fusion=True, device=device, amodel= 'HTSAT-tiny', tmodel='t5')
model.load_ckpt("./final_training_sk10-resume3/checkpoints/epoch_40.pt")

# Get the class index dict
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]

# Get all the data
audio_files = sorted(glob.glob(esc50_test_dir + '*.flac', recursive=True))
print(len(audio_files))
json_files = sorted(glob.glob(esc50_test_dir + '*.json', recursive=True))
ground_truth_idx = []
count = 0
end_idx = 200
for jf in json_files:
    dict_key = ", ".join(json.load(open(jf))['tags'])
    if dict_key in class_index_dict:
        ground_truth_idx.append(class_index_dict[dict_key])
    else:
        class_index_dict[dict_key] = end_idx
        ground_truth_idx.append(end_idx)
        end_idx = end_idx + 1
        count = count + 1

print(f"Missed files : {count}")

batch_size = 32

class CustomDataset(Dataset):
    def __init__(self, audio_files, ground_truth):
        self.audio_files = audio_files
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], self.ground_truth[idx]

# Define a custom collate function to stack audio tensors into a batch
def custom_collate_fn(batch):
    audio_batch, ground_truth_batch = zip(*batch)
    return audio_batch, ground_truth_batch

# Create DataLoader instances for the custom dataset
custom_dataset = CustomDataset(audio_files, ground_truth_idx)
custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

print(f"Len of loader : {len(custom_loader.dataset)}")

preds = None
for audio_batch, ground_truth_batch in (custom_loader):
    if audio_batch == None:
        break
    with torch.no_grad():
        # Pass the batches through your model for text and audio embeddings
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