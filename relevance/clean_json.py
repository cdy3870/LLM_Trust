import json
import re
import string
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm

# Download only on first run
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Preprocessing functions
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

# Load JSON
with open('./data/full_data.json', 'r') as f:
    data = json.load(f)

# Loop through the full structure
for complexity_level in tqdm(data):  # simple, complex, ...
    for model_name in data[complexity_level]:  # llama, gemma, ...
        list_of_lists = data[complexity_level][model_name]
        for inner_list in list_of_lists:  # <-- intermediate list
            for item in inner_list:  # <-- dictionaries containing full, paragraphs, sentences
                # full
                full_content, score = item["full"]
                if detect_english(full_content):
                    item["full"][0] = clean_text(full_content)
                else:
                    item["full"][0] = ''

                # paragraphs & sentences
                for section in ["paragraphs", "sentences"]:
                    cleaned = []
                    for content, score in item[section]:
                        if detect_english(content):
                            cleaned.append([clean_text(content), score])
                        else:
                            cleaned.append(['', score])
                    item[section] = cleaned

# Save
with open('./data/full_data_clean.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Saved full_data_clean.json")
