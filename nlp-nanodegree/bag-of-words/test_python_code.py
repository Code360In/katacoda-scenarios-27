import string
from nltk.stem import WordNetLemmatizer

text = ''

def remove_punctuation(text):
    for punctuation in string.punctuation: 
        text = text.replace(punctuation, ' ') 
    return text

no_punct = remove_punctuation(text) 

def lowercase (text): 
    lowercased = text.lower() 
    return lowercased

norm_case = lowercase(no_punct) #Remember, apply function to stepbystep

from nltk import word_tokenize 

def tokenize (text):
    tokenized = word_tokenize(text)
    return tokenized

tok_text = tokenize(norm_case)


def remove_numbers (text):
    words_only = [word for word in text if word.isalpha()]
    return words_only

no_num = remove_numbers(tok_text)


from nltk.corpus import stopwords 

# Create a list of english stopwords
stop_words = set(stopwords.words('english')) 

# Create function
def remove_stopwords (text):
    without_stopwords = [word for word in text if not word in stop_words]
    return without_stopwords

no_stop = remove_stopwords(no_num)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    return lemmatized

clean_text = lemmatize(no_stop)

def check_dict():
    # we could check length of dict, order
    pass

def check_output_solution_1():
    pass