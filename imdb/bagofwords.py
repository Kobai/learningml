import pandas as pd
df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter = "\t", quoting=3)
from bs4 import BeautifulSoup as soup
import re
import nltk
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer

def review_to_words(raw_review):
    # Remove html 
    text = soup(raw_review, "html5lib").get_text()

    # Use regular expressions to do a find-and-replace
    letters = re.sub("[^a-zA-Z]", " ", text)

    # All lower case and tokenize
    words = letters.lower().split()

    # Get rid of stop words
    words = [w for w in words if not w in stopwords.words("english")] 

    return(" ".join(words)) 


num_reviews = df['review'].size
clean_reviews = []

for i in range(0, num_reviews):
    if((i+1) % 1000 == 0):
        print((i+1))
    clean_reviews.append(review_to_words(df['review'][i]))


vectorizer = CountVectorizer(analyzer = "word",tokenizer=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_reviews)
train_data_features = train_data_features.toarray()

print(train_data_features.shape)

