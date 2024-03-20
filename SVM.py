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
from sklearn.feature_extraction.text import CountVectorizer

reviews = df['Review'].fillna('').tolist()
cv = CountVectorizer()
X = cv.fit_transform(reviews)
df_transformed = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())
y = df['Label'] 
  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X = cv.fit_transform(reviews)
cv = CountVectorizer()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
import numpy as np

# Mencari indeks baris dengan nilai NaN
nan_rows = np.isnan(X_train.toarray()).any(axis=1)

# Mengambil indeks baris yang tidak mengandung NaN
valid_rows = ~nan_rows

# Menggunakan indeks untuk memperoleh matriks tanpa baris yang mengandung NaN
X_train_no_nan = X_train[valid_rows]
y_train_no_nan = y_train[valid_rows]
for c in [0.01, 0.05, 0.25, 0.5, 1]:

    sv = SVC(C=c)
    sv.fit(X_train, y_train)
    print('Accuracy for C=%s: %s'
     %(c, accuracy_score(y_test, sv.predict(X_test))))
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Nilai C yang akan diuji


# Memprediksi label pada data uji
y_pred = sv.predict(X_test)

 # Mengukur dan mencetak metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print('Metrics for C=%s:' % c)
print('  Accuracy: %.3f' % accuracy)
print('  Precision: %.3f' % precision)
print('  Recall: %.3f' % recall)
print()

final_model_sv = SVC(C=1)
final_model_sv.fit(X, y)
print('Final Model Accuracy: %s' %accuracy_score(y_test, final_model_sv.predict(X_test)))