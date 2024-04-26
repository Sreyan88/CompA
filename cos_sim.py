import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description='A simple command-line program.')
parser.add_argument('--start', '-s', type=int)
parser.add_argument('--end', '-e', type=int)
args = parser.parse_args()

start_idx = args.start
end_idx = args.end

meta_df = pd.read_csv(".final_audioset_val.csv", header=0)

your_file_path = "././sim_embeddings_strong/sim_embeddings_strong_val0_16300.pkl"
embeddings_list = pickle.load(open(your_file_path, "rb"))

num_embeddings = len(embeddings_list)
print(f"Num embeddings: {num_embeddings}")

cosine_similarities = {}

end_idx = min(end_idx,num_embeddings)

for i in tqdm(range(start_idx, end_idx), unit="Embedding", total=(end_idx-start_idx)):
    curr_sim = {}
    for j in range(num_embeddings):
        cos_sim = F.cosine_similarity(embeddings_list[i].squeeze(0), embeddings_list[j].squeeze(0), dim=0)
        curr_sim[meta_df.iloc[j]["path"]] = cos_sim.detach().numpy().item()
    cosine_similarities[meta_df.iloc[i]["path"]] = curr_sim

with open("././sim_embeddings_strong/cos_sim_val_" + str(start_idx) + "_" + str(end_idx) + ".pkl", "wb") as file:
    pickle.dump(cosine_similarities, file)