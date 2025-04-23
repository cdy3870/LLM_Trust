import json


with open('granularity_data/full_data_4_trust.json', "r") as f1:
    for line1 in f1:
        d1 = json.loads(line1)
with open('granularity_data/full_data_4_relevancy.json', 'r') as file:
    d2 = json.load(file)
    

for k, v in d1.items():
    d1[k]["llama8b-r"] = d2[k]["llama8b-r"]
    # print(d2[k]["llama8b-r"])
    # break
    d1[k]["gemma9b-r"] = d2[k]["gemma9b-r"]

with open('granularity_data/full_data_4_combined.json', 'w') as f3:
    json.dump(d1, f3)

