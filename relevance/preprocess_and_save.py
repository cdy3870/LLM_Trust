import pandas as pd
import re
import string
import swifter
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')


# Helper functions
def detect_english(text):
    if not isinstance(text, str):
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def remove_punctuation(s):
    return s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

def remove_stop_words(input_str):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_str)
    return ' '.join([i for i in tokens if i not in stop_words])

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_string(input_str):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(input_str)
    return ' '.join([lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in tokens])

def clean_text(s):
    s = s.lower()
    s = re.sub(r'\d+', '', s)
    s = remove_punctuation(s)
    s = s.strip()
    s = ' '.join(s.split())
    s = remove_stop_words(s)
    s = re.sub('[^A-Za-z0-9 ]+', '', s)
    s = lemmatize_string(s)
    return s

# Load data
df = pd.read_csv("./data/combined_raw.csv")
print(f"Original count: {len(df)}")

# Keep only English rows
df = df[df["text"].swifter.apply(detect_english)]
print(f"English only: {len(df)}")

# Preprocess
df["text"] = df["text"].swifter.apply(clean_text)
df.dropna(subset=["text"], inplace=True)
df = df[df["text"] != ""]
print(f"After cleaning: {len(df)}")

# Save
df.to_csv("./data/combined_clean.csv", index=False)
print("✅ Saved sample_clean.csv")
