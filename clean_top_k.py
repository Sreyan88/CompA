import pickle

with open("./top_5/final_top_k.pkl", "rb") as f:
    top_dict = pickle.load(f)

csv_dict = pickle.load(open("./csv_dict.pkl", "rb"))

print(f"Len of top_dict : {len(top_dict)}")

for key, val in top_dict.items():
    if key in top_dict[key]:
        del top_dict[key][key]
    for k1, v1 in val.items():
        if k1 in csv_dict.keys() and len(csv_dict[k1]["caption"].split(";"))!=4:
            print(f"Path: {k1}, Len : {len(csv_dict[k1]['caption'].split(';'))} and Caption : {csv_dict[k1]['caption']}")

print(f"Len of top_dict : {len(top_dict)}")