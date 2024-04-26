import os
import torch
import numpy as np
import pickle
from tqdm.auto import tqdm
import pandas as pd

# Set the path to your folder containing the .pt files
folder_path = '././top_5/'

# Initialize an empty list to store the loaded embeddings
k = 5

top_k = {}

# Loop through the .pt files in the folder and load each file
for filename in sorted(os.listdir(folder_path)):
    if filename.startswith("cos_sim_"):
        top_5_sim_mtx = {}
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            print(f"Loading pickled file : {file_path}")
            data = pickle.load(f)
            print(f"Loaded pickled file : {file_path}")
        top_k.update(data)

with open(f"././top_5/final_top_k.pkl", "wb") as file:
    pickle.dump(top_k, file)