import pandas as pd
# 1. Read Data
df = pd.read_csv('data.csv', encoding = "ISO-8859-1",header=None)
df = df[:1000]
df.columns = ['target', 'tweet_id', 'datetime', 'query', 'username', 'tweet']
# print(df.target.unique())
df.replace({'target':{4:1}}, inplace = True)
# print(df.target.unique())
df.head(10)
print(df)


# Stemming
## Ngubah kata menjadi kata dasar -> Running -> run
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()

def remove_urls_and_tags(text, replacement_text=""):
    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+|@\S+')
 
    # Use the sub() method to replace URLs with the specified replacement text
    text_without_urls = url_pattern.sub(replacement_text, text)
 
    return text_without_urls

def stemming(content):
    content = remove_urls_and_tags(content)
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)   # the regular expression matches any pattern that is not a character
                                                          # (since negation ^ is used) and replaces those matched sequences 
                                                          # with empty space, thus all special characters and digits get 
                                                          # removed.
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]   
                                                          # apply port_stem only on words not in the list of stop-words
    stemmed_content = " ".join(stemmed_content)
    
    return stemmed_content

df['stemmed_content'] = df['tweet'].apply(stemming)
print(df)


# Ubah kata menjadi vecto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = df["stemmed_content"].values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


vectorizer = TfidfVectorizer()


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# print(X_train)
# print(X_test)

import pickle
import os 
os.makedirs("data", exist_ok=True)
df.to_csv('data/processed.csv')

with open("data/vectorizer.pickle", "wb") as f:
    pickle.dump(vectorizer, f)
    
train_data = {"x":X_train, "y":y_train}
test_data = {"x":X_test, "y":y_test}

with open("data/train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)
    
with open("data/test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)