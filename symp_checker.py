import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('Training.csv')
data

data.drop('Unnamed: 133', axis=1, inplace=True)
data['prognosis'].value_counts()
data['prognosis'].value_counts().count()
data.isna().sum()


symptom_count = data.apply(lambda x: True
if x['itching'] == 1 else False, axis=1)

num_rows = len(symptom_count[symptom_count == True].index)

symtom_dict = {}
for index, column in enumerate(data.columns):
    symtom_dict[column] = index

data['prognosis'].replace({}, inplace=True)
data

Y = data['prognosis']
X = data.drop('prognosis', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(f"Training set size {X_train.shape[0]}")
print(X_test.shape, Y_test.shape)
print(f"Validation set size {X_test.shape[0]}")

from sklearn.svm import SVC
rf=SVC(kernel='rbf',random_state=0)
rf = rf.fit(X_train, Y_train)
confidence = rf.score(X_test, Y_test)
print(f"Training Accuracy {confidence}")
Y_pred = rf.predict(X_test)
print(f"Validation Prediction {Y_pred}")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Validation accuracy {accuracy}")
conf_mat = confusion_matrix(Y_test, Y_pred)
print(f"confusion matrix {conf_mat}")
clf_report = classification_report(Y_test, Y_pred)

score = cross_val_score(rf, X_test, Y_test, cv=3)
print(score)

result = rf.predict(X_test)
accuracy = accuracy_score(Y_test, result)
clf_report = classification_report(Y_test, result)
print(f"accuracy {accuracy}")