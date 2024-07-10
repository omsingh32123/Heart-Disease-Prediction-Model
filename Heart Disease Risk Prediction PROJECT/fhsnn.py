# Import required libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# Load the dataset
df = pd.read_csv('framingham.csv')
# Drop rows with missing values
df.dropna(inplace=True)
# Split the dataset into features and target
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Multi-Layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)
# Fit the model on the training data
mlp.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = mlp.predict(X_test)
# Calculate the F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score: ", f1)
