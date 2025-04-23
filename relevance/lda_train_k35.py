import warnings
warnings.filterwarnings("ignore")

def train_lda_model(k, corpus, dictionary):
    import gensim
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=42,
        passes=10,
        eval_every=None
    )
    return lda_model

def main():
    import pandas as pd
    import gensim
    from gensim import corpora
    from gensim.models import CoherenceModel

    df = pd.read_csv("./data/combined_clean.csv")
    texts = df["text"].dropna().tolist()
    processed_texts = [doc.split() for doc in texts]
    print(len(processed_texts))

    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    print("corpus generated")

    k = 35
    lda_model = train_lda_model(k, corpus, dictionary)
    print("model generated")

    cm = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    score = cm.get_coherence()
    print(f"✅ Trained LDA with {k} topics — Coherence (c_v): {score:.4f}")

    # Save model
    lda_model.save("./models/lda_model_k35.gensim")
    print("💾 Model saved to lda_model_k35.gensim")

if __name__ == "__main__":
    main()
