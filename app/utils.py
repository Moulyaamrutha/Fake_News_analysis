import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

url_re = re.compile(r"http\S+|www\S+")
non_alphanum = re.compile(r"[^a-zA-Z0-9\s]")


def clean_text(text):
    text = text.lower()
    text = url_re.sub("", text)
    text = non_alphanum.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [lemm.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def keras_prepare_sequence(tokenizer, texts, max_len=128):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post")
