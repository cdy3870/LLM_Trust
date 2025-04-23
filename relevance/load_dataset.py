from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split='train[:1%]', trust_remote_code=True)
wiki_df = pd.DataFrame(wiki['text'], columns=['text'])

# ArXiv
arxiv = load_dataset("ccdv/arxiv-classification", split="train[:1%]")
arxiv_df = pd.DataFrame(arxiv['text'], columns=['text'])

# 20 Newsgroups
newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
newsgroup_df = pd.DataFrame(newsgroup.data, columns=['text'])

# Combine and save
df = pd.concat([wiki_df, arxiv_df, newsgroup_df], ignore_index=True)
df.dropna(subset=["text"], inplace=True)
df.to_csv("./data/combined_raw.csv", index=False)
print("✅ Saved combined_raw.csv")
