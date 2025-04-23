import json

# Load files
with open("./data/full_data.json", "r") as f:
    original_data = json.load(f)

with open("./data/full_data_scored.json", "r") as f:
    scored_data = json.load(f)

# Replace scores across all top-level keys
for top_key in scored_data:
    for model in scored_data[top_key]:
        for i, topic_group in enumerate(scored_data[top_key][model]):
            for j, topic_entry in enumerate(topic_group):
                original_entry = original_data[top_key][model][i][j]

                # Replace full score
                original_entry["full"][1] = topic_entry["full"][1]

                # Replace paragraph scores
                for k in range(len(topic_entry["paragraphs"])):
                    original_entry["paragraphs"][k][1] = topic_entry["paragraphs"][k][1]

                # Replace sentence scores
                for k in range(len(topic_entry["sentences"])):
                    original_entry["sentences"][k][1] = topic_entry["sentences"][k][1]

# Save updated version
with open("./data/full_data_updated.json", "w") as f:
    json.dump(original_data, f, indent=2)

print("✅ All scores updated and saved to full_data_updated.json")
