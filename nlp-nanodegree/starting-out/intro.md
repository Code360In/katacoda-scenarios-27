# Starting Out

## Prerequisites that were installed



````
pip install pandas-profiling==2.7.1
pip install contractions
pip install s3fs
sudo apt-get install tree


%time
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import sklearn
import itertools
import unicodedata

import string
import re
import nltk
import contractions
from bs4 import BeautifulSoup
````

