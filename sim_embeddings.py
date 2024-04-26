#!/usr/bin/env python

from transformers import AutoFeatureExtractor, ClapModel
import torch
import pandas as pd
from transformers import AutoTokenizer, ClapTextModelWithProjection
from transformers import ClapAudioModelWithProjection, ClapProcessor
from tqdm import tqdm
import librosa
import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='A simple command-line program.')
parser.add_argument('--start', '-s', type=int)
parser.add_argument('--end', '-e', type=int)
args = parser.parse_args()

start_idx = args.start
end_idx = args.end

model = ClapModel.from_pretrained("laion/clap-htsat-fused")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-fused")

meta_df = pd.read_csv(".final_audioset_val.csv", header=0)

text_model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-fused")
audio_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")

processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

meta_df = meta_df.iloc[start_idx:end_idx].copy()

audio_emds = []

os.makedirs("././sim_embeddings_strong_val", exist_ok=True)

device="cuda"
audio_model.to(device)
with open("error_files.txt", "w") as file:
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        audio_path = row['path']
        audio, sr = librosa.load(audio_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        # print(f"In index: {idx}")
        with torch.no_grad():
            try:
                inputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000)
                input_ftrs = inputs['input_features'].to(device)
                is_longer = inputs['is_longer']
                
                outputs = audio_model(input_features=input_ftrs, is_longer=is_longer)
                
                audio_embeds = outputs.audio_embeds.cpu()
                audio_emds.append(audio_embeds)
            except Exception as e:
                file.write(audio_path)
                file.write(os.linesep)

with open('././sim_embeddings_strong/sim_embeddings_strong_val' + str(start_idx) + "_" + str(end_idx) + '.pkl', "wb") as file:
    pickle.dump(audio_emds, file)
# torch.save(audio_emds, 'sim_embeddings_strong/sim_embeddings_strong_' + str(start_idx) + "_" + str(end_idx) + '.pkl')
# inputs = feature_extractor(random_audio, return_tensors="pt")
# audio_features = model.get_audio_features(**inputs)