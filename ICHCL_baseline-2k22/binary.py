import pandas as pd
import numpy as np
from glob import glob
import re
import json
import sys 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import stemmer as hindi_stemmer

def tr_flatten(d,lb):
    flat_text = []
    flat_text.append({
        'tweet_id':d['tweet_id'],
        'text':d['tweet'],
        'binary_label':lb[d['tweet_id']],
    })

    for i in d['comments']:
            flat_text.append({
                'tweet_id':i['tweet_id'],
                'text':flat_text[0]['text'] +' '+i['tweet'], #flattening comments(appending one after the other)
                'binary_label':lb[i['tweet_id']],
            })
            if 'replies' in i.keys():
                for j in i['replies']:
                    flat_text.append({
                        'tweet_id':j['tweet_id'],
                        'text':flat_text[0]['text'] +' '+ i['tweet'] +' '+ j['tweet'], #flattening replies
                        'binary_label':lb[j['tweet_id']],
                    })
    return flat_text

def te_flatten(d):
    flat_text = []
    flat_text.append({
        'tweet_id':d['tweet_id'],
        'text':d['tweet'],
    })

    for i in d['comments']:
            flat_text.append({
                'tweet_id':i['tweet_id'],
                'text':flat_text[0]['text'] + i['tweet'],
            })
            if 'replies' in i.keys():
                for j in i['replies']:
                    flat_text.append({
                        'tweet_id':j['tweet_id'],
                        'text':flat_text[0]['text'] + i['tweet'] + j['tweet'],
                    })
    return flat_text

regex_for_english_hindi_emojis="[^a-zA-Z#\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF\u0900-\u097F]"
def clean_tweet(tweet, english_stemmer, stopword):
    tweet = re.sub(r"@[A-Za-z0-9]+",' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+",' ', tweet)
    tweet = re.sub(regex_for_english_hindi_emojis,' ', tweet)
    tweet = re.sub("RT ", " ", tweet)
    tweet = re.sub("\n", " ", tweet)
    tweet = re.sub(r" +", " ", tweet)
    tokens = []
    for token in tweet.split():
        if token not in stopword:
            token = english_stemmer.stem(token)
            token = hindi_stemmer.hi_stem(token)
            tokens.append(token)
    return " ".join(tokens)

def main(argv):
    
    english_stopwords = stopwords.words("english")

    with open('final_stopwords.txt', encoding = 'utf-8') as f:
        hindi_stopwords = f.readlines()
        for i in range(len(hindi_stopwords)):
            hindi_stopwords[i] = re.sub('\n','',hindi_stopwords[i])
    
    stopword = english_stopwords + hindi_stopwords
    english_stemmer = SnowballStemmer("english")
    
    base_addreess = argv[0]
    directories = []
    for i in glob(base_addreess+"/Train/*/*/"):
        for j in glob(i+'*/'):
            directories.append(j)
    data = []
    for i in directories:
        try:
            with open(i+'data.json', encoding='utf-8') as f:
                data.append(json.load(f))
        except:
            continue
    binary_labels = []
    for i in directories:
        if('Hinglish' in i):
            with open(i+'binary_labels.json', encoding='utf-8') as f:
                binary_labels.append(json.load(f))
        else:
            try:
                with open(i+'labels.json', encoding='utf-8') as f:
                    binary_labels.append(json.load(f))
            except:
                continue
    
    data_label = []
    for i in range(len(binary_labels)):
        for j in tr_flatten(data[i], binary_labels[i]):
            data_label.append(j)
    
    train_len = len(data_label)
    df = pd.DataFrame(data_label, columns = data_label[0].keys(), index = None)
    df.loc[df['binary_label']=='NONE']='NOT'
    print("Binary Distribution")
    print(df['binary_label'].value_counts())
    
    tweets = df.text
    binary_y = df.binary_label 
    
    cleaned_tweets = [clean_tweet(tweet, english_stemmer, stopword) for tweet in tweets]
    
    vectorizer = TfidfVectorizer(min_df = 5)
    
    X = vectorizer.fit_transform(cleaned_tweets)
    X = X.todense()
    
    X_train, X_val, y_train, y_val = train_test_split(X, binary_y, test_size=0.2, random_state=42)
    classifier = KNeighborsClassifier(5)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_val)
    print("With K-Nearest Neighbour:")
    print(classification_report(y_val, y_pred))
    
    le = LabelEncoder() #label encoding labels for training Dense Neural Network
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    
    model = Sequential(
        [
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    
    model.compile('adam', loss='binary_crossentropy', metrics = ['accuracy']) #compiling a neural network with 3 layers for classification
    model.fit(X_train, y_train, epochs = 5, batch_size = 32)
    
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5).astype('int64')
    y_pred = y_pred.reshape(len(y_pred))    
    
    print("With MLP:")
    
    print(classification_report(y_val, y_pred))  

if __name__ == '__main__':
    main(sys.argv[1:])