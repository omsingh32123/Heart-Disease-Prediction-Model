# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('framingham.csv')

# Replace missing values with the mean
imp = SimpleImputer(strategy='mean')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

# Split the dataset into features and target
X = df.drop(['TenYearCHD'], axis=1)
y = df['TenYearCHD']

# Initialize the decision tree classifier with default hyperparameters
dtc = DecisionTreeClassifier()

# Fit the model on the entire dataset
dtc.fit(X, y)

# Create a new dataframe with all the features used to train the model
new_data = pd.DataFrame(columns=X.columns)
new_data.loc[0] = X.mean()

# Take user input for the features
print('Please provide the following details:')
age = float(input('Age: '))
gender = float(input('Gender (0 for female, 1 for male): '))
sysBP = float(input('Systolic Blood Pressure (mmHg): '))
diaBP = float(input('Diastolic Blood Pressure (mmHg): '))
BMI = float(input('Body Mass Index (kg/m^2): '))
heartRate = float(input('Resting Heart Rate (bpm): '))
glucose = float(input('Glucose (mg/dL): '))

# Update the new dataframe with user input
new_data.at[0, 'age'] = age
new_data.at[0, 'male'] = gender
new_data.at[0, 'sysBP'] = sysBP
new_data.at[0, 'diaBP'] = diaBP
new_data.at[0, 'BMI'] = BMI
new_data.at[0, 'heartRate'] = heartRate
new_data.at[0, 'glucose'] = glucose

# Make predictions on the new data
prediction = dtc.predict(new_data)
probability = dtc.predict_proba(new_data)[:,1]

# Print the results
if prediction == 0:
    print('You are not likely to have heart disease.')
else:
    print('You are likely to have heart disease.')
print('Probability of having heart disease:', probability[0])
