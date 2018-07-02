import pandas as pd
from bs4 import BeautifulSoup as soup
import re
from nltk.corpus import stopwords
import nltk.data
import logging

train = pd.read_csv("labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    review_text = soup(review, "html5lib").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

        return(words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Split paragrpah into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(
                raw_sentence, remove_stopwords))

    return sentences


sentences = []
for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)
print(len(sentences))


# Build model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Initialize various parameters
num_featues = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

# Initialize and train the model
from gensim.models import word2vec














