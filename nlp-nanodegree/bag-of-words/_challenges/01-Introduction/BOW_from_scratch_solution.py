import nltk  
import numpy as np  
import sklearn
import string
import pandas as pd
import bs4 as bs  
import urllib.request  
import re
import sys

print("Python version:", sys.version)
print("Version info.:", sys.version_info)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("skearn version:", sklearn.__version__)
print("re version:", re.__version__)
print("nltk version:", nltk.__version__)

# # Getting the data
urls = ['https://en.wikipedia.org/wiki/Natural_language_processing', 'https://en.wikipedia.org/wiki/Bag_of_words', 'https://en.wikipedia.org/wiki/Computer_vision']
def scrape_articles(urls):
    article_text = ''
    sentences = []
    for url in urls:
        raw_html = urllib.request.urlopen(url)  
        raw_html = raw_html.read()
        article_html = bs.BeautifulSoup(raw_html, 'lxml')
        article_paragraphs = article_html.find_all('p')
        for para in article_paragraphs:  
            article_text += para.text
        sentences.append(article_text)
    return sentences

sentences = scrape_articles(urls)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import string

def clean (sentences):
    clean_sentences = []
    for text in sentences:
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ') # Remove Punctuation
            
        lowercased = text.lower() # Lower Case
        
        tokenized = word_tokenize(lowercased) # Tokenize
        
        words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
        
        stop_words = set(stopwords.words('english')) # Make stopword list
        
        without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
        
        lemma=WordNetLemmatizer() # Initiate Lemmatizer
        
        lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
        clean_sentences.append(" ".join(lemmatized))
    return clean_sentences

clean_sentences = clean(sentences)

tokenized_sentences = [sub.split() for sub in clean_sentences]

wordfreq = {}

for tok_sen in tokenized_sentences:
    for token in tok_sen:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

import heapq # 
most_freq = heapq.nlargest(25, wordfreq, key=wordfreq.get)

sentence_vectors = []
for sentence in tokenized_sentences:
    sent_vec = []
    for token in most_freq:
        if token in sentence:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
print(f"Essential features extracted from corpus: {most_freq}")
print("\n")
sentence_vectors = np.asarray(sentence_vectors)
print(f"Matrix view of vectorized sentences: {sentence_vectors}")

