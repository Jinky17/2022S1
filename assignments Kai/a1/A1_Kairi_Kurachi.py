import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from itertools import chain