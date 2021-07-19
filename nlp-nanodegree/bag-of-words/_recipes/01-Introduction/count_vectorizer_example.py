# Raw text to analyse
sentences = ["All the better to see with my child, said the wolf.", 
"My! big teeth you have got! cried Little Red Riding Hood.",
"All the better to eat you up with, said the wolf."]

# Clean the text
# 1. remove punctuation
import re
sentences = [re.sub(r'[^A-Za-z]', ' ', s) for s in sentences]

# 2. lowercase
sentences = [s.lower() for s in sentences]

# 3. tokenize
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokenized_sentences = [word_tokenize(s) for s in sentences]

# 4. remove stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
clean_sentences = []
for tok_sen in tokenized_sentences:
    stop_rem = [word for word in tok_sen if str(word).lower() not in stop]
    clean_sentences.append(stop_rem)

# 5. Transform clean tokens back into clean sentences
output = []
for sent in clean_sentences:
    output.append(' '.join(sent))

# 6 Text transformation to vectorization
# Create an object
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# Generate the output for the Bag Of Words
BOW = cv.fit_transform(output)

# All words, including their index to model
print("Feature names:")
print(cv.get_feature_names())
print("\n")

# Show the output
import pandas as pd
count_vect_df = pd.DataFrame(BOW.todense(), columns=cv.get_feature_names())
pd.concat([count_vect_df, count_vect_df], axis=1)
print("Count vectorizer output:")
print("\n")
print(count_vect_df)