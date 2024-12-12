import sys
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
import random as rd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#####
#####

#Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("en-80k.txt", term_index=0, count_index=1)
# Correct word
def correct_text(text):
    sug = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    if sug:
        return sug[0].term
    else:
        return text


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
    words = [correct_text(word) for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


analyzer = SentimentIntensityAnalyzer()
# Calculate sentiment rate of a text
def get_sentiment_rate(text):
    scores = analyzer.polarity_scores(text)
    return np.abs(scores['compound'])


football_words = ["full time", "goal", "half time", "kick off", "owngoal", "penalty", "match", "red card", "yellow card"]
# Calculate the number of football words in a tweet
def count_football_words(text):
    return sum(word in text for word in football_words)


MatchIDs = [11, 12, 10, 14, 8, 2, 5, 13, 4, 17, 0, 19, 7, 3, 18, 1]
MatchnbIDs = [130, 97, 175, 130, 130, 130, 130, 130, 170, 130, 130, 130, 130, 130, 130, 110]
# Split correctement les données
def split_custom(X, y, test_size=0.3, random_state=42):
    random_chooser = rd.Random()
    random_chooser.seed(random_state)  # La valeur de seed permet de reproduire les mêmes choix

    test_choice = random_chooser.sample(MatchIDs, int(test_size*len(MatchIDs)))
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    index = 0
    for i in range(len(MatchIDs)):
        if index>=len(X):break
        for j in range(MatchnbIDs[i]):
            if index>=len(X):break
            if MatchIDs[i] not in test_choice:
                X_train.append(X[index])
                y_train.append(y[index])
            else:
                X_test.append(X[index])
                y_test.append(y[index])
            index+=1
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#####
#####

print("PREPROCESS PART 0...")
sys.stdout.flush()


os.makedirs("tmp/", exist_ok = True)
os.makedirs("tmp_kaggle/", exist_ok = True)


# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')

# Load GloVe model with Gensim's API
cond = False
cond = cond or not os.path.isfile("tmp/tweet_df.npy")
cond = cond or not os.path.isfile("tmp_kaggle/tweet_df_kaggle_GermanyGhana32.csv.npy")
cond = cond or not os.path.isfile("tmp_kaggle/tweet_df_kaggle_GermanySerbia2010.csv.npy")
cond = cond or not os.path.isfile("tmp_kaggle/tweet_df_kaggle_GreeceIvoryCoast44.csv.npy")
cond = cond or not os.path.isfile("tmp_kaggle/tweet_df_kaggle_NetherlandsMexico64.csv.npy")
if cond:
    embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings


print("PREPROCESS PART 0 : OK")
sys.stdout.flush()

#####
#####

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

#####
#####

print("PREPROCESS PART 2...")
sys.stdout.flush()


go = False
vector_size = 200  # Adjust based on the chosen GloVe model


if go or not os.path.isfile("tmp/X.npy") or not os.path.isfile("tmp/y.npy"):
    
    if go or not os.path.isfile("tmp/tweet_df.npy"):
        # Calcul la 1ère partie de l'embeddings
        tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
        tweet_df = pd.DataFrame(tweet_vectors)
        np.save("tmp/tweet_df.npy", tweet_df)
    else:
        tweet_df = np.load("tmp/tweet_df.npy")
        tweet_df = pd.DataFrame(tweet_df)

    print("tweet_df : OK")
    sys.stdout.flush()
    
    # Join les df
    period_features = pd.concat([df, tweet_df], axis=1)

    print("period_features : OK")
    sys.stdout.flush()


    # Ajouter une colonne contenant le nombre de tweets par PeriodID
    period_features['TweetCount'] = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('size').fillna(0)
    period_features['TweetCount'] = period_features['TweetCount'] / period_features['TweetCount'].max()
    
    # Ajouter une colonne contenant le nombre de mots liés au foot par tweet
    period_features['FootballWordCount'] = period_features['Tweet'].apply(count_football_words).fillna(0)
    period_features['FootballWordCount'] = period_features['FootballWordCount'] / period_features['FootballWordCount'].max()
    
    # Ajouter une colonne contenant le score de sentiment
    period_features['Sentiment'] = period_features['Tweet'].apply(get_sentiment_rate).fillna(0)

    print("add colonnes : OK")
    sys.stdout.flush()

    
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

    print("X-y : OK")
    sys.stdout.flush()
else:
    X = np.load("tmp/X.npy")
    y = np.load("tmp/y.npy")
    


print("PREPROCESS PART 2 : OK")
sys.stdout.flush()

#####
#####

# TESTING :

# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle

X_train, X_test, y_train, y_test = split_custom(X, y, test_size=0.3, random_state=42)

# SOLO :

clf =  LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("LR-1000 : ", accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("RF-100 : ", accuracy_score(y_test, y_pred))


clf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("RF-170 (10, sqrt) : ", accuracy_score(y_test, y_pred))


clf = XGBClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("XGB-100 : ", accuracy_score(y_test, y_pred))


clf = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("XGB-170 (0.1, 6, 1, logloss, gbtree) : ", accuracy_score(y_test, y_pred))

# MULTI :

rf = RandomForestClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(random_state=42, n_estimators=100)
lr = LogisticRegression(random_state=42, max_iter=1000)
svc =  SVC(kernel="rbf", random_state=42)
base_models = [ ('rf', rf), ('xgb', xgb), ('lr', lr), ('svc', svc) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-BASIC (rf, xgb, lr, svc) : ", accuracy_score(y_test, y_pred))

rf = RandomForestClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(random_state=42, n_estimators=100)
lr = LogisticRegression(random_state=42, max_iter=1000)
base_models = [ ('rf', rf), ('xgb', xgb), ('lr', lr) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-BASIC (rf, xgb, lr) : ", accuracy_score(y_test, y_pred))


rf = RandomForestClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(random_state=42, n_estimators=100)
base_models = [ ('rf', rf), ('xgb', xgb) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-BASIC (rf, xgb) : ", accuracy_score(y_test, y_pred))


rf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
xgb = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree")
lr = LogisticRegression(random_state=42, max_iter=1000)
svc =  SVC(kernel="rbf", random_state=42)
base_models = [ ('rf', rf), ('xgb', xgb), ('lr', lr), ('svc', svc) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-ADVANCED (rf, xgb, lr, svc) : ", accuracy_score(y_test, y_pred))

rf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
xgb = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree")
lr = LogisticRegression(random_state=42, max_iter=1000)
base_models = [ ('rf', rf), ('xgb', xgb), ('lr', lr) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-ADVANCED (rf, xgb, lr) : ", accuracy_score(y_test, y_pred))


rf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
xgb = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree")
base_models = [ ('rf', rf), ('xgb', xgb) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-ADVANCED (rf, xgb) : ", accuracy_score(y_test, y_pred))

# BOOTSTRAP :

clf_aux = RandomForestClassifier(random_state=42, n_estimators=100)
clf = BaggingClassifier(clf_aux, n_estimators=10, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("BOOTSTRAP-RF-100 : ", accuracy_score(y_test, y_pred))


clf_aux = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
clf = BaggingClassifier(clf_aux, n_estimators=10, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("BOOTSTRAP-RF-170 (10, sqrt) : ", accuracy_score(y_test, y_pred))

clf_aux = XGBClassifier(random_state=42, n_estimators=100)
clf = BaggingClassifier(clf_aux, n_estimators=10, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("BOOTSTRAP-XGB-100 : ", accuracy_score(y_test, y_pred))

clf_aux = XGBClassifier(random_state=42, n_estimators=170, learning_rate=0.1, max_depth=6, subsample=1, eval_metric="logloss", booster="gbtree")
clf = BaggingClassifier(clf_aux, n_estimators=10, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("BOOTSTRAP-XGB-170 (0.1, 6, 1, logloss, gbtree) : ", accuracy_score(y_test, y_pred))


###
###

rf = RandomForestClassifier(random_state=42, n_estimators=170, max_depth=10, max_features="sqrt")
xgb = XGBClassifier(random_state=42, n_estimators=100)
lr = LogisticRegression(random_state=42, max_iter=1000)
base_models = [ ('rf', rf), ('xgb', xgb), ('lr', lr) ]
meta_model = LogisticRegression()
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("STACK-HYBRIDE (rf, xgb, lr) : ", accuracy_score(y_test, y_pred))

#####
#####
#####
#####

print("KAGGLE 1...")
sys.stdout.flush()

go_kaggle = False


if go_kaggle or not os.path.isfile("predictions_rf_all.csv"):
    # This time we train our classifier on the full dataset that it is available to us.

    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X, y)
    predictions = []

    go = False
    vector_size = 200  # Adjust based on the chosen GloVe model

    # We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
    # Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
    # to be submitted on Kaggle.
    for fname in os.listdir("eval_tweets"):
        filename = "tmp_kaggle/preprocessing_" + fname + ".csv"
        
        if go or not os.path.isfile(filename):
            val_df = pd.read_csv("eval_tweets/" + fname)
            val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)
            val_df.to_csv(filename, index=False, encoding="utf-8")
        else:
            val_df = pd.read_csv(filename)

        ###
        filename = "tmp_kaggle/tweet_df_kaggle_" + fname + ".npy"

        if go or not os.path.isfile(filename):
            tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
            tweet_df = pd.DataFrame(tweet_vectors)
            np.save(filename, tweet_df)
        else:
            tweet_df = np.load(filename)
            tweet_df = pd.DataFrame(tweet_df)

        period_features = pd.concat([val_df, tweet_df], axis=1)
        ###

        ###
        period_features['TweetCount'] = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('size').fillna(0)
        period_features['TweetCount'] = period_features['TweetCount'] / period_features['TweetCount'].max()

        period_features['FootballWordCount'] = period_features['Tweet'].apply(count_football_words).fillna(0)
        period_features['FootballWordCount'] = period_features['FootballWordCount'] / period_features['FootballWordCount'].max()

        period_features['Sentiment'] = period_features['Tweet'].apply(get_sentiment_rate).fillna(0)
        ###

        period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
        period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
        X_pred = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

        preds = clf.predict(X_pred)
        period_features['EventType'] = preds
        predictions.append(period_features[['ID', 'EventType']])

    pred_df = pd.concat(predictions)
    pred_df.to_csv('predictions_glove_rf_all.csv', index=False)

print("KAGGLE 1 : OK")
sys.stdout.flush()

#####
#####

print("KAGGLE 2...")
sys.stdout.flush()

go_kaggle = False


if go_kaggle or not os.path.isfile("predictions_xgb_all.csv"):
    # This time we train our classifier on the full dataset that it is available to us.

    clf = XGBClassifier(random_state=42, n_estimators=100)
    clf.fit(X, y)
    predictions = []
    
    go = False
    vector_size = 200  # Adjust based on the chosen GloVe model

    # We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
    # Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
    # to be submitted on Kaggle.
    for fname in os.listdir("eval_tweets"):
        filename = "tmp_kaggle/preprocessing_" + fname + ".csv"
        
        if go or not os.path.isfile(filename):
            val_df = pd.read_csv("eval_tweets/" + fname)
            val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)
            val_df.to_csv(filename, index=False, encoding="utf-8")
        else:
            val_df = pd.read_csv(filename)

        ###
        filename = "tmp_kaggle/tweet_df_kaggle_" + fname + ".npy"

        if go or not os.path.isfile(filename):
            tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
            tweet_df = pd.DataFrame(tweet_vectors)
            np.save(filename, tweet_df)
        else:
            tweet_df = np.load(filename)
            tweet_df = pd.DataFrame(tweet_df)

        period_features = pd.concat([val_df, tweet_df], axis=1)
        ###

        ###
        period_features['TweetCount'] = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('size').fillna(0)
        period_features['TweetCount'] = period_features['TweetCount'] / period_features['TweetCount'].max()

        period_features['FootballWordCount'] = period_features['Tweet'].apply(count_football_words).fillna(0)
        period_features['FootballWordCount'] = period_features['FootballWordCount'] / period_features['FootballWordCount'].max()

        period_features['Sentiment'] = period_features['Tweet'].apply(get_sentiment_rate).fillna(0)
        ###

        period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
        period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
        X_pred = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

        preds = clf.predict(X_pred)
        period_features['EventType'] = preds
        predictions.append(period_features[['ID', 'EventType']])

    pred_df = pd.concat(predictions)
    pred_df.to_csv('predictions_glove_xgb_all.csv', index=False)

print("KAGGLE 2 : OK")
sys.stdout.flush()