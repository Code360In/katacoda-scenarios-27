# Bag of Words model from scratch

We have seen previously that the CountVectorizer function can perform the vectorization for the BOW model with only a few lines of code. Let us walk through each step of the process, this time through the coded implementation. 

## Prerequisites

It is advisable to [setup a Python virtual environment](https://docs.python.org/3/library/venv.html) so that the interpreter, libraries, and scripts installed are isolated from other environments on your operating system.

```python
python3 -m venv /path/to/new/virtual/environment
```

Our input in previous examples consisted of only a few sentences. We will scrape a real-world example on a Wikipedia article by using the urllib and BeautifulSoup packages which, respectively, make an HTTP request for an article and parses the HTML returned from that request. The parsed document can then be searched through for all text available, and analysis can then be performed.

We will need the following packages.
```python
import nltk  
import numpy as np  
import random  
import string

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