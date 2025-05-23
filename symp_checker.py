
# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing scikit-learn modules for model training and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('Training.csv')
data # Display the dataset

# Drop unnecessary column (Unnamed: 133) which is not useful
data.drop('Unnamed: 133', axis=1, inplace=True)

# Checking the distribution of prognosis (target variable)
data['prognosis'].value_counts()

data['prognosis'].value_counts().count() # Checking the number of unique prognosis classes
data.isna().sum() # Checking for missing (NaN) values


# Counting number of rows where 'itching' symptom is present
symptom_count = data.apply(lambda x: True
if x['itching'] == 1 else False, axis=1)
num_rows = len(symptom_count[symptom_count == True].index)

# Creating a dictionary that maps symptom names to column indices
symtom_dict = {}
for index, column in enumerate(data.columns):
    symtom_dict[column] = index

# Separate features (X) and target (Y)
Y = data['prognosis']
X = data.drop('prognosis', axis=1)

# Splitting dataset into training and testing sets (67% train, 33% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(f"Training set size {X_train.shape[0]}")
print(X_test.shape, Y_test.shape)
print(f"Validation set size {X_test.shape[0]}")

# Train an SVM (Support Vector Machine) model
from sklearn.svm import SVC
svm_model=SVC(kernel='rbf',random_state=0, probability=True)
svm_model = svm_model.fit(X_train, Y_train)

# Evaluate training accuracy
confidence = svm_model.score(X_test, Y_test)
print(f"Training Accuracy {confidence}")

critical_symptoms = ['chest_pain', 'shortness_of_breath', 'severe_headache', 'unconsciousness']
# Make predictions on validation set
Y_pred = svm_model.predict(X_test)
print(f"Validation Prediction {Y_pred}")

# Calculate validation accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Validation accuracy {accuracy}")

# Generate confusion matrix
conf_mat = confusion_matrix(Y_test, Y_pred)
print(f"confusion matrix {conf_mat}")
clf_report = classification_report(Y_test, Y_pred)

# Perform cross-validation with 3 folds
score = cross_val_score(svm_model, X_test, Y_test, cv=3)
print(score)

# Final model prediction and accuracy
result = svm_model.predict(X_test)
accuracy = accuracy_score(Y_test, result)
clf_report = classification_report(Y_test, result)
print(f"accuracy {accuracy}")

# Adding user interaction to input symptoms and predict the disease

# Function to preprocess user input and make a prediction
def predict_disease(user_symptoms):
    # Normalize input symptoms (strip, lowercase, and replace spaces with underscores)
    user_symptoms = [symptom.strip().lower().replace(" ", "_") for symptom in user_symptoms]
    input_data = np.zeros(len(X.columns))  # X.columns has the feature names


    # Map symptoms to corresponding feature indices
    for symptom in user_symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)  # Get the index of the symptom in the feature set
            input_data[index] = 1  # Set the corresponding index to 1 (symptom present)

# Convert input data to a DataFrame (same format as the model input)
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Predict the disease
    prediction = svm_model.predict(input_df)[0]
    
    print("\n Based on the symptoms you provided, the predicted disease is:", prediction)

   # List of critical symptoms that need urgent care


# Get symptoms from the user
def get_user_input():
    print("\nPlease enter symptoms separated by commas (e.g., chest_pain, shortness_of_breath):")
    user_input = input("Enter your symptoms: ")
    user_symptoms = user_input.split(",")  # Split by commas to create a list of symptoms
    return user_symptoms

# Check for critical symptoms
def check_critical_symptoms(user_symptoms):
    urgent = [symptom for symptom in user_symptoms if symptom in critical_symptoms]
    if urgent:
        print("\n  WARNING: Critical symptom(s) detected:", ', '.join(urgent))
        print("Please seek immediate medical attention.\n")
    else:
        print(f"Please note: The symptoms '{user_symptoms}' you entered is not critical and you can find other symptoms below.")
        print("Available symptoms are:")
        available_symptoms = [symptom.replace('_', ' ').title() for symptom in X.columns]
        for s in available_symptoms:
            print(f"- {s}")



user_symptoms = get_user_input()  # Get symptoms from the user
predict_disease(user_symptoms)  # Predict the disease
check_critical_symptoms(user_symptoms)


