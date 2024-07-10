import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load the dataset
df = pd.read_csv('framingham.csv')

# Preprocess the data
df = df.dropna()
X = df.drop('TenYearCHD', axis=1) # Features
y = df['TenYearCHD'] # Target variable
X = pd.get_dummies(X, columns=['education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']) # Convert categorical variables to binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Train the logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = lr.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
