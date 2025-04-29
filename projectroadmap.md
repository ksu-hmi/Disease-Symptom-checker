Sprint 1:

1. Submit project topic for approval under assignment 6 on d2l. ✅ Done
2. Post a brief information about the project on Teams Projects Spreadsheet. ✅ Done
3. Welcome people who are working on same topic to join my team. ✅ Done
4. Found a related project and a working code an described below what it does: => ✅ Done

   
               1. Imported Libraries:              
numpy (np): Supports numerical operations, array handling, and mathematical functions.

pandas (pd): Used for data manipulation and analysis; essential for loading the dataset and handling data frames.

seaborn (sns): Data visualization library; useful for making statistical graphics and visualizing relationships within the data.

matplotlib.pyplot (plt): Plotting library; creates static, animated, and interactive visualizations.

sklearn.model_selection.train_test_split:  Splits the dataset into training and testing sets.

sklearn.model_selection.cross_val_score: Performs cross-validation to evaluate the model's performance across multiple subsets of the data.

sklearn.metrics.accuracy_score: Calculates the model's accuracy by comparing true and predicted labels.

sklearn.metrics.confusion_matrix: Builds a confusion matrix to visualize prediction errors.

sklearn.metrics.classification_report: Generates a report that includes precision, recall, F1-score, and support metrics.

sklearn.ensemble.RandomForestClassifier: (Initially imported) Used for training Random Forest models; however, not used in the final training — replaced by SVC.

pickle: Enables saving (serializing) trained machine learning models to a file for later use.
               
                2. Data Collection:
File Used: Training.csv
Action: Loaded the dataset containing symptoms and corresponding disease prognosis labels.

                3. Data Cleaning and Preprocessing:
Removed unnecessary columns (Unnamed: 133).
Checked for missing values and ensured data integrity.
Created a dictionary mapping each symptom to its corresponding index for easier feature handling.
                                        
                4. Feature Engineering  => Done              
Input Features (X): All symptom columns.
Output Target (Y): prognosis column (disease labels).

                5. Dataset Splitting:        
Split the data into training and testing sets:
Training set: 67%
Validation set: 33%
Random State: 42 (for reproducibility).

                6. Model Development: 
Algorithm Used: Support Vector Classifier (SVC) with an RBF kernel.  
Initially imported RandomForestClassifier but ultimately used SVC for better performance.  
Trained the SVC model using the training dataset.

                7. Model Evaluation:
Calculated:
    Training Accuracy
    Validation Accuracy

Evaluated performance using:
    Confusion Matrix
    Classification Report
    Cross-Validation Score (3-fold)


5. Edit the Readme.md file => ✅ Done
6. Create a Project Roadmap file =>✅ Done

Sprint 2:

1. Add  comments to the original code to make it more readable and understandable =>✅ Done
2. Add user interaction: Allow users to manually input their symptoms. =>✅ Done
3. Created predict_disease(user_symptoms) function =>✅ Done

Converts user-provided symptoms into the model's input format.

Predicts the disease based on input symptoms using the trained SVM model.

4. Critical symptoms detection:  =>✅ Done
   
Introduced a list of critical symptoms (e.g., chest_pain, shortness_of_breath, severe_headache, unconsciousness).

Checks if user input contains any critical symptoms and warns the user to seek immediate medical attention.


5. Input normalization:  =>✅ Done

Processes user input by converting symptoms to lowercase and replacing spaces with underscores to match feature names in the dataset.

6. User feedback improvements:  =>✅ Done

Added warning messages if unrecognized symptoms are entered.

Listed available valid symptoms when invalid input is detected.

