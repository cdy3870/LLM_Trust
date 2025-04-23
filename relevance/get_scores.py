import json
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from tqdm import tqdm

# Load model and dictionary
lda_model = LdaModel.load("./models/lda_model_k35.gensim")
dictionary = Dictionary.load("./models/lda_model_k35.gensim.id2word")

# Load data
with open("./data/full_data_clean.json", "r") as f:
    data = json.load(f)

# Helper function
def avg_similarity_to_topic(tokens, topic_id):
    word_ids = [dictionary.token2id[word] for word in tokens if word in dictionary.token2id]
    topic_terms = dict(lda_model.get_topic_terms(topic_id, topn=len(dictionary)))
    weights = [topic_terms[word_id] for word_id in word_ids if word_id in topic_terms]
    return sum(weights) / len(weights) if weights else 0.0

# Process with tqdm through all layers
for complexity in tqdm(data, desc="Data"):
    for model in tqdm(data[complexity], desc=f"Complexity Levels ({complexity})", leave=False):
        for inner_list in tqdm(data[complexity][model], desc=f"Models ({model})", leave=False):
            for item in tqdm(inner_list, desc="Items", leave=False):
                topic_tokens = item["topic"].split()
                topic_bow = dictionary.doc2bow(topic_tokens)
                topic_distribution = lda_model.get_document_topics(topic_bow)
                if not topic_distribution:
                    continue
                best_topic_id = max(topic_distribution, key=lambda x: x[1])[0]

                for section in ["full", "paragraphs", "sentences"]:
                    if section == "full":
                        tokens = item[section][0].split()
                        score = avg_similarity_to_topic(tokens, best_topic_id)
                        item[section][1] = score
                    else:
                        new_entries = []
                        for content, _ in item[section]:
                            tokens = content.split()
                            score = avg_similarity_to_topic(tokens, best_topic_id)
                            new_entries.append([content, score])
                        item[section] = new_entries

# Save
with open("./data/full_data_scored.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Saved full_data_scored.json")
