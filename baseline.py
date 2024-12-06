###### IMPORT


import sys
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


###### UTILS FUNCTIONS

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

analyzer = SentimentIntensityAnalyzer()
# Calculate sentiment rate of a text
def get_sentiment_rate(text):
    scores = analyzer.polarity_scores(text)
    return np.abs(scores['compound'])

football_words = ["goal", "penalty", "match", "red card", "yellow card"]
# Calculate the number of football words in a tweet
def count_football_words(text):
    return sum(word in tweet for word in football_words)


###### PREPROCESS PART 0


print("PREPROCESS PART 0...")
sys.stdout.flush()


os.makedirs("tmp/", exist_ok = True)


# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')

# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings


print("PREPROCESS PART 0 : OK")
sys.stdout.flush()


###### PREPROCESS PART 1


print("PREPROCESS PART 1...")
sys.stdout.flush()


go = False


if go or not os.path.isfile("tmp/processing1.csv"):
    # Read all training files and concatenate them into one dataframe
    li = []
    for filename in os.listdir("train_tweets"):
        df = pd.read_csv("train_tweets/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Apply preprocessing to each tweet
    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    df.to_csv("tmp/processing1.csv", index=False, encoding="utf-8")
else:
    df = pd.read_csv("tmp/processing1.csv")


print("PREPROCESS PART 1 : OK")
sys.stdout.flush()


###### PREPROCESS PART 2


print("PREPROCESS PART 2...")
sys.stdout.flush()


vector_size = 200  # Adjust based on the chosen GloVe model
go = True


if go or not os.path.isfile("tmp/X.npy") and not os.path.isfile("tmp/y.npy"):
    # Apply preprocessing to each tweet and obtain vectors
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # Attach the vectors into the original dataframe
    period_features = pd.concat([df, tweet_df], axis=1)

    ##
    ##
    
    # Ajouter une colonne contenant le nombre de tweets par PeriodID
    period_features['TweetCount'] = period_features.groupby('PeriodID')['Tweet'].transform('size').fillna(0)
    period_features['TweetCount'] = period_features['TweetCount'] / period_features['TweetCount'].max()

    # Ajouter une colonne contenant le nombre de mots liés au foot par tweet
    period_features['FootballWordCount'] = period_features['Tweet'].apply(count_football_words).fillna(0)
    period_features['FootballWordCount'] = period_features['FootballWordCount'] / period_features['FootballWordCount'].max()

    print(period_features)
    sys.stdout_flush()

    # Ajouter une colonne contenant le score de sentiment
    period_features['Sentiment'] = period_features['Tweet'].apply(get_sentiment_rate).fillna(0)
    
    ##
    ##
    
    # Drop the columns that are not useful anymore
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    
    # Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    # We drop the non-numerical features and keep the embeddings values for each period
    X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
    # We extract the labels of our training samples
    y = period_features['EventType'].values

    np.save("tmp/X.npy", X)
    np.save("tmp/y.npy", y)
else:
    X = np.load("tmp/X.npy")
    y = np.load("tmp/y.npy")


print("PREPROCESS PART 2 : OK")
sys.stdout.flush()


###### EVALUATING ON A TEST SET


# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We set up a basic classifier that we train and then calculate the accuracy on our test set
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(random_state=42, n_estimators=100).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features='sqrt').fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = XGBClassifier(random_state=42, n_estimators=170, eval_metric="logloss").fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree").fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = SVC(kernel="rbf", random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = SVC(C=0.5, kernel="poly", degree=7, random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree").fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

clf = XGBClassifier(random_state=42, n_estimators=195, learning_rate=0.2, max_depth=3, subsample=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
                    
print("Test set: ", accuracy_score(y_test, y_pred))

clf = XGBClassifier(random_state=42, n_estimators=195, learning_rate=0.2, max_depth=3, subsample=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
                    
print("Test set: ", accuracy_score(y_test, y_pred))


###### For Kaggle submission


print("KAGGLE...")
sys.stdout.flush()


# This time we train our classifier on the full dataset that it is available to us.

clf = XGBClassifier(random_state=42, n_estimators=140, learning_rate=0.2, max_depth=3, subsample=1)
clf.fit(X, y)
predictions = []


# We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# to be submitted on Kaggle.
for fname in os.listdir("eval_tweets"):
    val_df = pd.read_csv("eval_tweets/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)

    ###
    period_features['TweetCount'] = period_features.groupby('PeriodID')['Tweet'].transform('size')
    ###
    
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X_pred = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds = clf.predict(X_pred)
    period_features['EventType'] = preds
    predictions.append(period_features[['ID', 'EventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('predictions.csv', index=False)


print("KAGGLE : OK")
sys.stdout.flush()
