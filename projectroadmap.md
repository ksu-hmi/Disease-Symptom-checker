1. Found a related project and a working code              
                                        1. Data Collection   => Done 
File Used: Training.csv
Action: Loaded the dataset containing symptoms and corresponding disease prognosis labels.

                                        2. Data Cleaning and Preprocessing
Removed unnecessary columns (Unnamed: 133).
Checked for missing values and ensured data integrity.
Created a dictionary mapping each symptom to its corresponding index for easier feature handling.
                                        3. Feature Engineering  => Done 
                                        
Input Features (X): All symptom columns.
Output Target (Y): prognosis column (disease labels).

                                        4. Dataset Splitting  => Done           
Split the data into training and testing sets:
Training set: 67%
Validation set: 33%
Random State: 42 (for reproducibility).

                                        5. Model Development  => Done 
Algorithm Used: Support Vector Classifier (SVC) with an RBF kernel.  
Initially imported RandomForestClassifier but ultimately used SVC for better performance.  
Trained the SVC model using the training dataset.

                                        6. Model Evaluation  => Done 
Calculated:
    Training Accuracy
    Validation Accuracy

Evaluated performance using:
    Confusion Matrix
    Classification Report
    Cross-Validation Score (3-fold)

                                        7. Future Enhancements (Planned)  => Done 
Implement a user interface for easier symptom input.
Integrate trusted medical databases (e.g., PubMed, ICD-10 codes).
Explore more advanced models like Random Forests, Gradient Boosting, or Deep Neural Networks.
Expand dataset with additional symptoms and diseases for better generalization.
