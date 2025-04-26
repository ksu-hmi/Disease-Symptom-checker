1. Found a related project and a working code an described below what it does: 


               1. Imported Libraries => Done              
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
               
                2. Data Collection   => Done 
File Used: Training.csv
Action: Loaded the dataset containing symptoms and corresponding disease prognosis labels.

                3. Data Cleaning and Preprocessing
Removed unnecessary columns (Unnamed: 133).
Checked for missing values and ensured data integrity.
Created a dictionary mapping each symptom to its corresponding index for easier feature handling.
                                        
                4. Feature Engineering  => Done              
Input Features (X): All symptom columns.
Output Target (Y): prognosis column (disease labels).

                5. Dataset Splitting  => Done           
Split the data into training and testing sets:
Training set: 67%
Validation set: 33%
Random State: 42 (for reproducibility).

                6. Model Development  => Done 
Algorithm Used: Support Vector Classifier (SVC) with an RBF kernel.  
Initially imported RandomForestClassifier but ultimately used SVC for better performance.  
Trained the SVC model using the training dataset.

                7. Model Evaluation  => Done 
Calculated:
    Training Accuracy
    Validation Accuracy

Evaluated performance using:
    Confusion Matrix
    Classification Report
    Cross-Validation Score (3-fold)

                  8. Future Enhancements (Planned)  => Done 
Implement a user interface for easier symptom input.
Integrate trusted medical databases (e.g., PubMed, ICD-10 codes).
Explore more advanced models like Random Forests, Gradient Boosting, or Deep Neural Networks.
Expand dataset with additional symptoms and diseases for better generalization.
