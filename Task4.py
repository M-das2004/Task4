import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay

# Load the dataset
# The file is accessible via its unique content fetch ID.
df = pd.read_csv('uploaded:data.csv')

# Display the first few rows to understand the data structure
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()

# 2. Train/test split and standardize features.
# The 'diagnosis' column is our target variable.
# The 'id' column is an identifier and should be dropped.
# All other columns are features.

# Prepare the data
# Drop the 'id' column as it's not a feature for the model.
df = df.drop('id', axis=1)

# Separate features (X) and target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Convert the categorical target variable to numerical (M=1, B=0)
y = y.apply(lambda x: 1 if x == 'M' else 0)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
# Standardization is a critical step for many machine learning models,
# including logistic regression, as it makes all features have the same scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit a Logistic Regression model.
# Instantiate and train the model.
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = model.predict(X_test_scaled)

# 4. Evaluate with confusion matrix, precision, recall, ROC-AUC.
print("\n--- Model Evaluation ---")

# Confusion Matrix
# This shows the number of correct and incorrect predictions made by the model.
# The rows represent the actual classes (0=Benign, 1=Malignant), and the columns represent the predicted classes.
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
# This provides precision, recall, and F1-score for each class.
# Precision: Of all the positive predictions, how many were actually positive?
# Recall: Of all the actual positives, how many did we predict correctly?
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# ROC-AUC Score
# The Area Under the Receiver Operating Characteristic Curve.
# It measures the model's ability to distinguish between the two classes.
# A score of 1.0 is a perfect classifier, while 0.5 is a random one.
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Plot the ROC curve for a visual representation of the trade-off
print("\nPlotting ROC Curve...")
disp = RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
disp.ax_.set_title('ROC Curve for Logistic Regression')
plt.show()


# 5. Tune threshold and explain sigmoid function.
# Logistic regression doesn't output a hard 0 or 1, but a probability.
# The sigmoid function, represented as $σ(z) = \frac{1}{1 + e^{-z}}$,
# transforms the linear combination of inputs into a probability between 0 and 1.
# By default, a probability of > 0.5 is classified as 1, and <= 0.5 is classified as 0.
# We can adjust this threshold.

print("\n--- Threshold Tuning Demonstration ---")

# Get the predicted probabilities for the positive class (Malignant)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Let's try a different threshold, for example, 0.3.
# A lower threshold makes the model more sensitive to predicting the positive class (Malignant).
new_threshold = 0.3
y_pred_new_threshold = (y_prob >= new_threshold).astype(int)

# Re-evaluate with the new threshold
conf_matrix_new = confusion_matrix(y_test, y_pred_new_threshold)
print(f"Confusion Matrix with a new threshold of {new_threshold}:")
print(conf_matrix_new)

# Notice how the values in the confusion matrix change.
# Lowering the threshold often increases the number of True Positives (higher recall)
# but might also increase the number of False Positives (lower precision).
print("\nClassification Report with a new threshold of 0.3:")
print(classification_report(y_test, y_pred_new_threshold, target_names=['Benign', 'Malignant']))
