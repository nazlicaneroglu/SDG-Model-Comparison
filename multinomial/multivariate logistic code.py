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
from sklearn.linear_model import Lasso
import numpy as np

file_path = "/Users/admin/Desktop/thesis/SDG-Model-Comparison/datasets/osdg-community-data-v2023-04-01.xlsx"
df = pd.read_excel(file_path)
print(df)
df['text'] = df['text'].str.lower()


keyword_data = pd.read_excel("/Users/admin/Desktop/thesis/SDG-Model-Comparison/datasets/remove_dup_keywords.xlsx")

#create dummy for sdg 1 
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


#For SDG1
sdg1_keywords = keyword_data[keyword_data['SDGs'] == "SDG1"]['Words (Phrases)'].tolist()

# For each SDG1 related keyword
for keyword in sdg1_keywords:
    # Count the occurrences of the keyword in the text column
    df[keyword] = df['text'].str.count(keyword.lower())
print(df.columns)

# List to store the columns with at least one "1" value
columns_with_ones = []

# Iterate through each keyword column
for keyword in sdg1_keywords:
    # Check if the keyword column has at least one "1" value
    if df[keyword].any():
        columns_with_ones.append(keyword)

# Print the columns with at least one "1" value
print(columns_with_ones)

# List to store the columns to delete
columns_to_delete = []

# Iterate through each keyword column
for keyword in sdg1_keywords:
    # Check if the keyword column exists in the DataFrame
    if keyword in df.columns:
        # Check if all values in the keyword column are "0"
        if df[keyword].sum() == 0:
            columns_to_delete.append(keyword)
    else:
        print(f"Column '{keyword}' does not exist.")

# Delete the columns with only "0" values
df = df.drop(columns=columns_to_delete)

# Print the remaining columns
print(df.columns)

# Assign the feature and target variables
features = df.iloc[:, -524:]
target = df['sdg_1_dummy']


# Check for perfect separation
# Uncomment the following line to check for perfect separation
print(target.value_counts())

# check for multicollinearity
correlation_matrix = features.corr()
print(correlation_matrix)

vif = pd.DataFrame()
vif["features"] = features.columns
vif["vif"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
print(vif)

# set the vif threshold
vif_threshold = 5  # adjust the threshold as needed

# get the variables with vif above the threshold
high_vif_vars = vif[vif["vif"] > vif_threshold]["features"]

# drop the variables with high vif from the feature set
features_filtered = features.drop(columns=high_vif_vars)

# print the remaining variables after dropping high vif variables
print("remaining variables after dropping high vif variables:")
print(features_filtered.columns)
vif.to_csv('vif_sdg1.csv', index=False)

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



