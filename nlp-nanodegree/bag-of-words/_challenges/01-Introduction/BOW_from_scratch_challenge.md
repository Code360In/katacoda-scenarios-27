# Bag of Words model from scratch

We have seen previously that the CountVectorizer function can perform the vectorization for the BOW model with only a few lines of code. Let us walk through each step of the process, this time through the coded implementation. 

## Objectives
0. Extract the data
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
import sys
import numpy as np  
import string
import bs4 as bs  
import urllib.request  
```
Check that your system includes the following packages, and the versions match the versions below.

```python
print("Python version:", sys.version)
print("numpy version:", np.__version__)
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

Let's create our corpus using Wikipedia articles. Using the function below, we will make the url request for Wikipedia articles on [Natural Langugae Processings](https://en.wikipedia.org/wiki/Natural_angugae_processing), [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) and [Computer vision](https://en.wikipedia.org/wiki/Computer_vision), extract the paragraph text, and return those strings in a list.

```python
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
print(f"The corpus consists of {len(sentences)} documents")
```

Now that we have the raw text extracted from the the articles, we can then begin our cleaning steps.

### Remove punctuation

```python
# Iterate through the list of sentences and remove punctuation
```

### Lowercase

```python
# Iterate through the list of sentences and normalize the case
```

### Remove stop words

```python
# Iterate through the list of sentences and remove stop words
```

### Remove numbers

```python
# Iterate through the list of sentences and remove numbers

```

### Lemmatize
Once we have stripped the corpus of punctuation and normalized case. removed the stop words, numbers and lemmatized the grammatical variation of the words  ('tanning', 'tans' -> 'tan), we are ready to build our vocabulary.

```python
# Iterate through the list of sentences and remove lemmatize the tokens

```

### Build vocabulary using a dictionary
We take the tokenized list of our cleaned corpus, and create a dictionary to store these words as the key and their corresponding frequency as the value.

```python
# Tokenize each string from list in the corpus

# Create a dictionary to store word and frequency pairs

# Iterate through each sentence list in the tokenized sentences

# Iterate through each word in the sentence list, if the word is not found in the dictionary keys, add to dictionary; if the word is found in the dictionary, increment the value count of its frequency by 1. 

```

### Filter the dictionary

We can use the following function to filter our dictionary to the top 20 most frequently occurring words

```python
import heapq # 
most_freq = heapq.nlargest(5, wordfreq, key=wordfreq.get)
```

### Vectorize

At our final stage in the BOW pipeline, we convert the corpus into its vector representation. 

```python
# Create a list object to store all the sentence vectors

# Iterate through each tokenized sentence and create a list object for each sentence vector

# Iterate through each token in the most_freq dictionary, if the word is not found in the dictionary keys, add to dictionary; if the word is found in the tokens for the sentence, append 1 to the individual sentence vector, otherwise append 0. 

# Add each of of the individual vector back to list object which stores all sentence vectors to represent as a matrix
```

### BOW model
You can see the BOW model, containing either 0 and 1, if that feature is found in each document.