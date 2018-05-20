import pandas as pd 
from bs4 import BeautifulSoup as soup
import re
from nltk.corpus import stopwords
import nltk.data


train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_wordlist(review, remove_stopwords=False):
    review_text = soup(review,"html5lib").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

        return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# def review_to_sentences(review, tokenizer, remove_stopwords-False):


