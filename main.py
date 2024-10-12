#Deployment script to run email against models
#Run the script with python3 main.py
#Pick option 1 through 4 to see your desired model prediction
#Option 0 to exit the program
from myFunc import deployModel, getDF, getFeaturesPreSplit, getFeaturesPostSplit, deployEnsemble

def mainProgram():
    try:
        for option in modelDict:
            print(f'[{option}]: {modelDict[option]}')
        print('Enter 0 to exit...')
        modelFile = input('Which model would you like to use?: ')
        if modelFile == '0': return 'exit'
        if modelFile == '1': return deployModel('LogisticRegression_KayCheng.pkl', inputFeature)
        if modelFile == '2': return deployModel('MLPClassifier_ZiHin.pkl', inputFeature)
        if modelFile == '3': return deployEnsemble(nbFeature)
        if modelFile == '4': return deployModel('XGBoost_sebastian.joblib', inputFeature)
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