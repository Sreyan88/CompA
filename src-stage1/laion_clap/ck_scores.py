import numpy as np
import torch
from hook import CLAP_Module
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

csv_path = ".annotation_with_triplet.csv"
# csv_path = ".noun_verb/final_noun_verb_benchmark.csv"

sound_path = ".annotation_sounds/"
# sound_path = ".noun_verb/nv_bench_files/"

# Specify the directory path
directory_path = './test-newfile-resume-2/checkpoints/'

# List all files in the directory
file_list = os.listdir(directory_path)

device = torch.device('cuda')

result_dict = {"file": [], "f_score": [], "g_score": [], "h_score": [], }

for filename in sorted(file_list):
    # Construct the full path to each file
    file_path = os.path.join(directory_path, filename)
    
    # Check if the item in the directory is a file (not a subdirectory)
    if os.path.isfile(file_path):
        model = CLAP_Module(enable_fusion=True, device=device, amodel= 'HTSAT-tiny', tmodel='t5')
        model.load_ckpt(file_path)
        # model.load_ckpt("./final_training_sk10-resume3/checkpoints/epoch_top_0.pt")
        print("Model loaded")
        # model.load_ckpt(".630k-best.pt")
        # model.load_ckpt()

        df = pd.read_csv(csv_path,header=0)

        # df = df.head(500)

        class CustomDataset(Dataset):
            def __init__(self, captions, rev_captions, paths, rev_paths):
                self.captions = captions
                self.rev_captions = rev_captions
                self.audio_files = paths
                self.rev_audio_files = rev_paths

            def __len__(self):
                return len(self.audio_files)

            def __getitem__(self, idx):
                return self.audio_files[idx], self.rev_audio_files[idx], self.captions[idx], self.rev_captions[idx]

        # Define a custom collate function to stack audio tensors into a batch
        def custom_collate_fn(batch):
            audio_batch, rev_audio_batch, caption_batch, rev_caption_batch = zip(*batch)
            return audio_batch, rev_audio_batch, caption_batch, rev_caption_batch

        batch_size = 32

        df["pair_file"] = df["pair_file"].apply(lambda x: sound_path + x)
        df["reversed_pair_file"] = df["reversed_pair_file"].apply(lambda x: sound_path + x)

        custom_dataset = CustomDataset(df["pair_caption"].tolist(), df["reversed_pair_caption"].tolist(),df["pair_file"].tolist(),df["reversed_pair_file"].tolist())
        custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        audio_emb_list = []
        text_emb_list = []

        rev_audio_emb_list = []
        rev_text_emb_list = []

        for audio_batch, rev_audio_batch, caption_batch, rev_caption_batch in (custom_loader):
            with torch.no_grad():
            # Pass the batches through your model for text and audio embeddings
                text_embed = model.get_text_embedding(caption_batch,use_tensor=True)
                audio_embed = model.get_audio_embedding_from_filelist(audio_batch,use_tensor=True)
                
                rev_text_embed = model.get_text_embedding(rev_caption_batch,use_tensor=True)
                rev_audio_embed = model.get_audio_embedding_from_filelist(rev_audio_batch,use_tensor=True)

                # print(f"Text embed shape : {text_embed.shape}")
                # print(f"Audio embed shape : {audio_embed.shape}")
                
                audio_emb_list = audio_emb_list + [audio_embed[i:i+1, :] for i in range(audio_embed.shape[0])]
                text_emb_list = text_emb_list + [text_embed[i:i+1, :] for i in range(text_embed.shape[0])]

                rev_audio_emb_list = rev_audio_emb_list + [rev_audio_embed[i:i+1, :] for i in range(rev_audio_embed.shape[0])]
                rev_text_emb_list = rev_text_emb_list + [rev_text_embed[i:i+1, :] for i in range(rev_text_embed.shape[0])]

        f_score = []
        g_score = []
        h_score = []

        for i in range(len(text_emb_list)):
            sim_0_0 = F.cosine_similarity(text_emb_list[i].squeeze(0), audio_emb_list[i].squeeze(0), dim=0).cpu().numpy().item()
            sim_0_1 = F.cosine_similarity(text_emb_list[i].squeeze(0), rev_audio_emb_list[i].squeeze(0), dim=0).cpu().numpy().item()
            sim_1_0 = F.cosine_similarity(rev_text_emb_list[i].squeeze(0), audio_emb_list[i].squeeze(0), dim=0).cpu().numpy().item()
            sim_1_1 = F.cosine_similarity(rev_text_emb_list[i].squeeze(0), rev_audio_emb_list[i].squeeze(0), dim=0).cpu().numpy().item()

            f_sc = 1 if (sim_0_0 > sim_1_0) and (sim_1_1 > sim_0_1) else 0
            g_sc = 1 if (sim_0_0 > sim_0_1) and (sim_1_1 > sim_1_0) else 0
            h_sc = 1 if (f_sc==1 and g_sc==1) else 0

            f_score.append(f_sc)
            g_score.append(g_sc)
            h_score.append(h_sc)

        print(f"Avg f_score : {sum(f_score)/len(f_score)}")
        print(f"Avg g_score : {sum(g_score)/len(g_score)}")
        print(f"Avg h_score : {sum(h_score)/len(h_score)}")

        result_dict["file"].append(filename)
        result_dict["f_score"].append(sum(f_score)/len(f_score))
        result_dict["g_score"].append(sum(g_score)/len(g_score))
        result_dict["h_score"].append(sum(h_score)/len(h_score))

df = pd.DataFrame.from_dict(result_dict)
df.to_csv("././latest_test-newfile-resume-2.csv",index=False)