# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:55:53 2023

@author: pc
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
import statsmodels.api as sm



file_path = "C:/Users/pc/Desktop/Thesıs/thesis codes/osdg-community-data-v2023-04-01.xlsx"
df = pd.read_excel(file_path)
print(df)

keyword_data = pd.read_excel("C:/Users/pc/Desktop/Thesıs/thesis codes/remove_dup_keywords.xlsx")

#create dummies for each sdg
df['sdg_1_dummy'] = (df['sdg'] == 1) & (df['labels_positive'] > df['labels_negative']) & (df['agreement'] > 0.75)

# Convert boolean values to 1 or 0
df['sdg_1_dummy'] = df['sdg_1_dummy'].astype(int)

# Display the updated DataFrame
print(df)

# Count the occurrences of new_dummy_variable = 1
count_sdg1 = df['sdg_1_dummy'].value_counts()[1]
count_sdg1_keywords = keyword_data['SDGs'].value_counts()['SDG1']

# Display the count
print("Count of sdg1 texts = 1:", count_sdg1) #1101
print("Count of sdg1 keywords = 1:", count_sdg1_keywords) #717

#create indicator for each sdg
df['text'] = df['text'].str.lower()

#For SDG1
sdg1_keywords = keyword_data[keyword_data['SDGs'] == "SDG1"]['Words (Phrases)'].tolist()

# Create a new column for each SDG1 related keyword in the main dataset
for keyword in sdg1_keywords:
    df[keyword] = df['text'].str.contains(keyword.lower()).astype(int)
print(df.columns)


# Extract the features (newly created keyword columns) and the target variable
features = df[sdg1_keywords]
target = df['sdg_1_dummy']

# Check for perfect separation
# Uncomment the following line to check for perfect separation
print(target.value_counts())

# Check for multicollinearity
# Uncomment the following lines to calculate correlation matrix or VIF
correlation_matrix = features.corr()
print(correlation_matrix)
vif = pd.DataFrame()
vif["Features"] = features.columns
vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
print(vif)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Add a constant column to the training features
X_train = sm.add_constant(X_train)

# Create an instance of Logistic Regression
logreg = sm.Logit(y_train, X_train)

# Fit the logistic regression model on the training data
result = logreg.fit(maxiter=1000)

# Add a constant column to the testing features
X_test = sm.add_constant(X_test)

# Get the predicted probabilities of the features on the testing data
predicted_probabilities = result.predict(X_test)

# Convert predicted probabilities to binary predictions (0 or 1) using a threshold of 0.5
predictions = (predicted_probabilities >= 0.5).astype(int)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

# Display the p-values
p_values = result.pvalues[1:]  # Exclude the constant term
p_values = p_values.rename(index=dict(zip(p_values.index, X_train.columns[1:])))  # Rename the index with feature names
print("\nP-values:")
print(p_values)

vif.to_csv('vif.csv', index=False)  # Set index=False if you don't want to save the DataFrame index
