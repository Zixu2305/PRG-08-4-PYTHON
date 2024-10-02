#Main.py will be where we do the master codes and the shared codes
#Please do read the README.md file to understand the project and steps to take

#Introducing libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pickle
from joblib import load
import myFunc

#Reading of dataset files
df = pd.read_csv('./Main/data/testSampleChatgpt.csv', encoding='ISO-8859-1')
dfClean = pd.read_csv('./Main/data/cleanedData.csv')
df = df.drop(columns=['Unnamed: 6'])
df = myFunc.cleanDataframe(df)

# Separate target(label) from predictor columns
y = df.label

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit_transform(dfClean['fullContent'])
tfidf_matrix = tfidf_vectorizer.transform(df['fullContent'])

# Continuous features normalization
scaler = StandardScaler()
contd = scaler.fit_transform(df[['punctuationCount', 'subjectLength', 'bodyLength', 'totalLength']])

# Sparse binary features
sparse_features = csr_matrix(df[["urls", "totalLength", "generalConsumer", "govDomain", "eduDomain", "orgDomain", "netDomain", "otherDomain", "html", "punctuationCount"]].values)

X = hstack([sparse_features, contd, tfidf_matrix])

with open('./Main/model/MLPClassifier_ZiHin.pkl', 'rb') as file: 
    mlpC = pickle.load(file)

mlpC_pred_prob = mlpC.predict_proba(X)
mlpC_pred = mlpC.predict(X)
print(mlpC_pred_prob)
print(mlpC_pred)