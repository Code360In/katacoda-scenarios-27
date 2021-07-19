import string
from nltk.stem import WordNetLemmatizer

text = ''

def remove_punctuation(text):
    for punctuation in string.punctuation: 
        text = text.replace(punctuation, ' ') 
    return text

stepbystep = remove_punctuation(text) 

def lowercase (text): 
    lowercased = text.lower() 
    return lowercased

stepbystep = lowercase(stepbystep) #Remember, apply function to stepbystep

from nltk import word_tokenize 

def tokenize (text):
    tokenized = word_tokenize(text)
    return tokenized

stepbystep = tokenize(stepbystep)


def remove_numbers (text):
    words_only = [word for word in text if word.isalpha()]
    return words_only

stepbystep = remove_numbers(stepbystep)


from nltk.corpus import stopwords 

# Create a list of english stopwords
stop_words = set(stopwords.words('english')) 

# Create function
def remove_stopwords (text):
    without_stopwords = [word for word in text if not word in stop_words]
    return without_stopwords

stepbystep = remove_stopwords(stepbystep)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    return lemmatized

stepbystep = lemmatize(stepbystep)