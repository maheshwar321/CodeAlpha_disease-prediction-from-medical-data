import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Ensure the correct path to your dataset
file_path = 'C:/Users/mahes/Downloads/first project of codealpha/historical_financial_data.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} does not exist. Please check the path.")

# Step 1: Data Preprocessing
# Load the dataset
try:
    data = pd.read_csv(file_path)
except Exception as e:
    raise IOError(f"An error occurred while reading the file: {e}")

# Display the first few rows and columns of the dataset
print(data.head())
print(data.columns)

# Adding a synthetic target column for demonstration purposes
import numpy as np
np.random.seed(42)
data['target'] = np.random.choice([0, 1], size=len(data))

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns
non_numeric_cols = data.select_dtypes(exclude=['number']).columns

# Handle missing values
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Development
# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Random Forest
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

# Step 4: Model Evaluation
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Logistic Regression Evaluation
logistic_eval = evaluate_model(y_test, y_pred_logistic)
print(f"Logistic Regression: Accuracy: {logistic_eval[0]:.2f}, Precision: {logistic_eval[1]:.2f}, Recall: {logistic_eval[2]:.2f}, F1-score: {logistic_eval[3]:.2f}, ROC-AUC: {logistic_eval[4]:.2f}")

# Decision Tree Evaluation
tree_eval = evaluate_model(y_test, y_pred_tree)
print(f"Decision Tree: Accuracy: {tree_eval[0]:.2f}, Precision: {tree_eval[1]:.2f}, Recall: {tree_eval[2]:.2f}, F1-score: {tree_eval[3]:.2f}, ROC-AUC: {tree_eval[4]:.2f}")

# Random Forest Evaluation
forest_eval = evaluate_model(y_test, y_pred_forest)
print(f"Random Forest: Accuracy: {forest_eval[0]:.2f}, Precision: {forest_eval[1]:.2f}, Recall: {forest_eval[2]:.2f}, F1-score: {forest_eval[3]:.2f}, ROC-AUC: {forest_eval[4]:.2f}")
