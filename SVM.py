import pandas as pd
import openpyxl
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_excel('Dataset_spotify.xlsx')
df.head()

from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def pre_process(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = text.strip()
    pisah = text.split()
    tokens = nltk.tokenize.word_tokenize(text)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    
    return text

df['Review'] = df['Review'].apply(lambda x:pre_process(x))
df.head()
    