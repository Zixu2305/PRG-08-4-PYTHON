def cleanDataframe(df):
    import warnings
    warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')
    df.fillna({'subject': 'no subject'}, inplace=True)
    df["emailDomain"]=df['sender'].str.split('@').str[1]
    publicEmailDomainRegex = r'^(gmail|hotmail|outlook|yahoo)\.com'
    govRegex = r'\.gov(\.|$)'
    eduRegex = r'\.edu(\.|$)'
    orgRegex = r'\.org(\.|$)'
    netRegex = r'\.net(\.|$)'
    htmlPattern = r'<[^>]+>'
    punctuationRegex = r'[!?@_]'
    cleanSpace = r'(^\w\s)'

    df['generalConsumer'] = df['emailDomain'].str.contains(publicEmailDomainRegex, regex=True, na=False).astype(int)
    df['govDomain'] = df['emailDomain'].str.contains(govRegex, regex=True, na=False).astype(int)
    df['eduDomain'] = df['emailDomain'].str.contains(eduRegex, regex=True, na=False).astype(int)
    df['orgDomain'] = df['emailDomain'].str.contains(orgRegex, regex=True, na=False).astype(int)
    df['netDomain'] = df['emailDomain'].str.contains(netRegex, regex=True, na=False).astype(int)
    df['otherDomain'] = df[['generalConsumer', 'govDomain', 'eduDomain', 'orgDomain', 'netDomain']].apply(lambda row: 1 if (row == 0).all() else 0, axis=1)
    df['html'] = df['body'].str.contains(htmlPattern, regex=True, na=False).astype(int)
    df['fullContent'] = df['subject'] + ' ' + df['body']
    df['fullContent'] = df['fullContent'].str.lower()
    df['fullContent'] = df['fullContent'].str.replace(cleanSpace,'')
    df['punctuationCount'] = df['body'].str.count(punctuationRegex)
    df['subjectLength'] = df['subject'].astype(str).apply(lambda x: len(x))
    df['bodyLength'] = df['body'].astype(str).apply(lambda x: len(x))
    df['totalLength'] = df['fullContent'].astype(str).apply(lambda x: len(x))
    return df

#get accuracy score and probability determined by each of the models 
def predictAccProba(multinomial, bernoulli, gaussian, dataDict):
    from sklearn.metrics import accuracy_score
    preds_mnb = multinomial.predict(dataDict['testText'])
    preds_bnb = bernoulli.predict(dataDict['testBinary'])
    preds_gnb = gaussian.predict(dataDict['testContinuous'])

    accM = accuracy_score(dataDict['testLabel'], preds_mnb)
    accB = accuracy_score(dataDict['testLabel'], preds_bnb)
    accG = accuracy_score(dataDict['testLabel'], preds_gnb)

    probaM = multinomial.predict_proba(dataDict['testText'])
    probaB = bernoulli.predict_proba(dataDict['testBinary'])
    probaG = gaussian.predict_proba(dataDict['testContinuous'])

    accList = [accM,accB,accG]
    probaList = [probaM,probaB,probaG]
    return accList, probaList

#calculate the weights of each model for soft voting as an ensemble
#the soft voting is done in respect of each models prediction made instead of hard voting that does majority win
def calculateWeight(accList):
    total_accuracy = sum(accList)
    print(f"Multinomial Accuracy: {accList[0]*100:.2f}%")
    print(f"Bernoulli Accuracy: {accList[1]*100:.2f}%")
    print(f"Gaussian Accuracy: {accList[2]*100:.2f}%")
    print()
    weights = []
    # Normalize to get weights
    for i in accList:
        weights.append(i / total_accuracy)

    # Print the weights
    print(f'Weight for Multinomial: {weights[0]:.2f}')
    print(f'Weight for Bernoulli: {weights[1]:.2f}')
    print(f'Weight for Gaussian: {weights[2]:.2f}')
    return weights

#Applies the weights to the probabilities which will determine the ensemble predictions
def printEnsembleAccuracy(weights, probabilities, testLabel):
    # Combine probabilities with weights
    combined_proba = (weights[0] * probabilities[0] +
                    weights[1] * probabilities[1] +
                    weights[2] * probabilities[2])

    # Make final predictions
    y_pred_final = np.argmax(combined_proba, axis=1)
    # Evaluate accuracy
    # By matching final prediction to test label, the sum of correct prediction / number of prediction made
    accuracy = (y_pred_final == testLabel.values).mean()
    print(f'Naive Bayes Ensemble Accuracy: {accuracy*100:.2f}%')
    return y_pred_final

#visualize the ensemble performance with a classification report and confusion matrix
def printCM(predictLabels, trueLabels):
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    class_report = classification_report(trueLabels, predictLabels)
    print(class_report)

    cm = confusion_matrix(trueLabels, predictLabels)

    # Define class labels for the confusion matrix (for binary classification)
    class_labels = ['Phishing', 'Non-Phishing']  # Adjust according to your classes

    # Create the heatmap without annotations (annot=False)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)

    # Set axis labels
    ax.set_xlabel('True', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)

    # Set the class labels on the axes
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)

    # Set title
    plt.title('Confusion Matrix')

    # Manually add True Positive, False Positive, True Negative, and False Negative annotations
    # You must use the exact cell positions for each term in the 2x2 confusion matrix
    ax.text(0.5, 0.5, 'TP\n(' + str(cm[1, 1]) + ')', ha='center', va='center', fontsize=12)
    ax.text(1.5, 0.5, 'FP\n(' + str(cm[0, 1]) + ')', ha='center', va='center', fontsize=12)
    ax.text(0.5, 1.5, 'FN\n(' + str(cm[1, 0]) + ')', ha='center', va='center', fontsize=12)
    ax.text(1.5, 1.5, 'TN\n(' + str(cm[0, 0]) + ')', ha='center', va='center', fontsize=12)

    # Show the plot
    plt.tight_layout()
    return plt.show()

def deployModel(modelName, xFeatures):
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    import pickle
    import joblib
    filePath = './model/'
    if '.pkl' in modelName:
        with open(filePath + modelName, 'rb') as file: 
            model = pickle.load(file)
    else:
        model = joblib.load(filePath + modelName)

    pred_prob = model.predict_proba(xFeatures)
    pred = model.predict(xFeatures)
    print(pred_prob)
    print(pred)    
    return