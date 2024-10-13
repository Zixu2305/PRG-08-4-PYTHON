This is a project repository for INF1002-Programming Fundementals Group P8-4.
Where we are tackling a problem on identifying phishing emails using machine learning classification models.

1. Installation of libraries that will be used in this project
```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost setuptools
```

2. What to run before model tuning
    1. Ensure all libraries required are installed
    2. If cleanedData.csv is not in data folder, run all cells in dataExploration.ipynb to generate file
    3. Proceed with training & tuning the model in the training folder
    4. Training file naming convention should be, >type of model<_>your name<.ipynb
    5. Save your model with pickle dump into the model folder

3. Dataset collections
CEAS_08.csv & SpamAssasin.csv
https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952
Citation
A. I. Champa, M. F. Rabbi, and M. F. Zibran, “Curated datasets and feature analysis for phishing email detection with machine learning,” in 3rd IEEE International Conference on Computing and Machine Intelligence (ICMI), 2024, pp. 1–7 (to appear)

Sebastian:
https://www.kaggle.com/code/stpeteishii/email-spam-prediction-xgboost

4. Folder Structure
```
.
└── Main/
    ├── data/
    │   └── CEAS_08.csv
    │   └── cleanedData.csv
    │   └── SpamAssasin.csv
    │   └── testSampleChatgpt.csv
    ├── model/
    │   └── bernoulliNB_zixu.pkl
    │   └── emsembleWeights.pkl
    │   └── gaussianNB_zixu.pkl
    │   └── LogisticRegression_KayCheng.pkl
    │   └── MLPClassifier_ZiHin.pkl
    │   └── multinomialNB_zixu.pkl
    │   └── randomforestclassifier_daniel.pkl
    │   └── XGBoost_sebastian.pkl
    ├── training/
    │   └── LogisticRegression_KayCheng.py
    │   └── MLPClassifier_ZiHin.py
    │   └── NaiveBayesEnsemble_ZiXu.py
    │   └── RandomForestClassifier.py
    │   └── XGBoost_Sebastian.py
    ├── dataExploration.ipynb
    ├── main.py
    ├── myFunc.py
    └── README.md
```
