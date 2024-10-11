#Main.py will be where we do the master codes and the shared codes
#Please do read the README.md file to understand the project and steps to take

#Introducing libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from myFunc import deployModel, getDF, getFeaturesPreSplit, getFeaturesPostSplit, deployEnsemble

def mainProgram():
    try:
        for option in modelDict:
            print(f'[{option}]: {modelDict[option]}')
        print('Enter 0 to exit...')
        modelFile = input('Which model would you like to use?: ')
        if modelFile == '0': return 'exit'
        if modelFile == '1': deployModel('LogisticRegression_KayCheng.pkl', inputFeature)
        if modelFile == '2': deployModel('MLPClassifier_ZiHin.pkl', inputFeature)
        if modelFile == '3': deployEnsemble(nbFeature)
        if modelFile == '4': deployModel('XGBoost_sebastian.joblib', inputFeature)
    except:
        return print('Wrong input, try again.')

modelDict = {
    '1': 'Logistic Regression',
    '2': 'MLP Classifier',
    '3': 'Naive Bayes Classifier Ensemble',
    '4': 'XGBoost Classifier'
}        
running = True
inputFeature = getFeaturesPreSplit(*getDF())
nbFeature = getFeaturesPostSplit(*getDF())
while running:
    state = mainProgram()
    print()
    if state == 'exit': break