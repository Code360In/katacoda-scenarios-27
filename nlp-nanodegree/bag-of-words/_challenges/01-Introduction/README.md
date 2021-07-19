# Bag of Words model from scratch

We have seen previously that the CountVectorizer function can perform the vectorization for the BOW model with only a few lines of code. Let us walk through each step of the process, this time through the coded implementation. 

## Objectives
1. Clean the text extracted from Wikipedia articles
2. Tokenize the cleaned corpus
3. Create a dictionary of the tokenized words
4. Convert the dictionary of our corpus into their vector representations
5. Return the matrix from our BOW model

## Prerequisites

It is advisable to [setup a Python virtual environment](https://docs.python.org/3/library/venv.html) so that the interpreter, libraries, and scripts installed are isolated from other environments.

```python
python3 -m venv /path/to/new/virtual/environment
```

We will need the following packages.
```python
import nltk  
import numpy as np  
import sklearn
import string
import pandas as pd
import bs4 as bs  
import urllib.request  
import re
import sys
```
Check that your system includes the following packages, and the versions match the versions below.

```python
print("Python version:", sys.version)
print("Version info.:", sys.version_info)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("skearn version:", sklearn.__version__)
print("re version:", re.__version__)
print("nltk version:", nltk.__version__)
```

Our input in previous examples consisted of only a few sentences. We will perform some topic classification and feature extraction from real-world data, using scraped articles from Wikipedia article. This can be done with the urllib and BeautifulSoup packages which, respectively, make an HTTP request for an article and parses the HTML returned from that request. The parsed document can then be searched through for all text available, in this case we use the 'p' tag (for paragraph text).

```python
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')  
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:  
    article_text += para.text
```

Above shows article text extracted from the Wikipedia article on Natural Langugae Preprocessing. Let's build our corpus with additional articles. Write a function below which makes th url request for the articles from Wikipedia on [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) and [Computer vision](https://en.wikipedia.org/wiki/Computer_vision), extracts the paragraph text, and stores those strings in a list.

```python
# Function which takes each raw_html request and returns the extracted paragraphs as strings in a list called documents
```



Now that we have the raw text extracted from the the articles, we can then begin our cleaning steps.

### Remove punctuation

```python
```

### Lowercase

```python
```

### Remove stop words

```python
```

### Remove numbers

```python
```

### Lemmatize

```python
```
Once we have stripped the corpus of punctation and normalized case. removed the stop words, numbers and lemmatized the grammatical variation of the words  ('tanning', 'tans' -> 'tan), we are ready to build our vocabulary.

### Tokenize
```python
```

### Build vocabulary using a dictionary
We take the tokenized list of our cleaned corpus, and create a dictionary to store these words as the key and their corresponding frequency as the value.

```python
# Tokenize the corpus

# Create a dictionary to store word and frequency pairs

# Iterate through each sentence in the corpus and tokenize into words

# Iterate through each word n the sentence, if the word is not found in the dictionary keys, add to dictionary; if the word is found inthe dictionary, increment the value count of its frequency by 1. 

```

### Filter the dictionary

We can use the following function to filter our dictionary to the top 20 most frequently occurring words
```
import heapq # 
most_freq = heapq.nlargest(5, wordfreq, key=wordfreq.get)
```

### Vectorize
At our final stage in the BOW pipeline, we convert the corpus into its vector representation. 
```python
# Create a list object to store the text vectorization


```