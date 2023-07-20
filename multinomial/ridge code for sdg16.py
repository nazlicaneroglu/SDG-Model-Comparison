
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
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import numpy as np
import re


file_path = "/Users/admin/Desktop/thesis/SDG-Model-Comparison/datasets/osdg-community-data-v2023-04-01.xlsx"
df = pd.read_excel(file_path)
print(df)
df['text'] = df['text'].str.lower()


keyword_data = pd.read_excel("/Users/admin/Desktop/thesis/SDG-Model-Comparison/datasets/remove_dup_keywords.xlsx")

#create dummy for sdg 1 
df['sdg_16_dummy'] = (df['sdg'] == 16) & (df['labels_positive'] > df['labels_negative']) & (df['agreement'] > 0.75)

# Convert boolean values to 1 or 0
df['sdg_16_dummy'] = df['sdg_16_dummy'].astype(int)

# Display the updated DataFrame
print(df)

# Count the occurrences of new_dummy_variable = 1
count_sdg16 = df['sdg_16_dummy'].value_counts()[1]
count_sdg16_keywords = keyword_data['SDGs'].value_counts()['SDG16']

# Display the count
print("Count of sdg16 texts = 1:", count_sdg16) #1101
print("Count of sdg16 keywords = 1:", count_sdg16_keywords) #717

#For SDG1
sdg16_keywords = keyword_data[keyword_data['SDGs'] == "SDG16"]['Words (Phrases)'].tolist()

for keyword in sdg16_keywords:
    # Escape the special characters in the keyword
    escaped_keyword = re.escape(keyword.lower())
    # Count the occurrences of the keyword in the text column
    df[keyword] = df['text'].str.count(escaped_keyword)
print(df.columns)

# Subset the DataFrame to include only rows with 'sdg_label' equals to 'sdg1'
df_sdg16 = df[df['sdg_16_dummy'] == 1]

# List to store the columns with at least one "1" value for texts labelled as sdg1
columns_with_ones_sdg16 = []

# Iterate through each keyword column
for keyword in sdg16_keywords:
    keyword = keyword.strip()  # Remove leading/trailing whitespaces
    # Check if the keyword column exists in the DataFrame and 
    # it has at least one "1" value for texts labelled as sdg1
    if keyword in df_sdg16.columns and df_sdg16[keyword].any():
        columns_with_ones_sdg16.append(keyword)

# Print the columns with at least one "1" value for texts labelled as sdg1
print(columns_with_ones_sdg16)

# List to store the columns to delete
columns_to_delete_sdg16 = []

# Iterate through each keyword column
for keyword in sdg16_keywords:
    keyword = keyword.strip()  # Remove leading/trailing whitespaces
    # Check if the keyword column exists in the DataFrame and 
    # it does NOT have any "1" value for texts labelled as sdg1
    if keyword in df.columns and not df.loc[df['sdg_16_dummy'] == 1, keyword].any():
        columns_to_delete_sdg16.append(keyword)

# Delete the columns with only "0" values for SDG1 labelled texts
df = df.drop(columns=columns_to_delete_sdg16)

# Print the remaining columns
print(df.columns)

# Assign the feature and target variables
features = df.iloc[:, -len(columns_with_ones_sdg16):]
target = df['sdg_16_dummy']
'''
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

# Assign the feature and target variables
features = features_filtered
target = df['sdg_1_dummy']

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

'''
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.linear_model import RidgeCV

# List of alphas to try 
alphas = np.logspace(1, 3, 50)  # Adjust the range and granularity as needed

# Create a RidgeCV model
ridgecv = RidgeCV(alphas=alphas, cv=5)  # 5-fold cross-validation

# Fit the model
ridgecv.fit(X_train, y_train)

# Get the alpha that was selected
best_alpha = ridgecv.alpha_

print("Best alpha:", best_alpha)
#Best alpha: 51.79474679231202

# Replace Logistic Regression with Ridge Regression
ridge = Ridge(alpha=best_alpha)  # Adjust the alpha parameter as needed

# Fit the Ridge regression model on the training data
ridge.fit(X_train, y_train)

# Get the predicted probabilities of the features on the testing data
predicted_probabilities = ridge.predict(X_test)

# Convert predicted probabilities to binary predictions (0 or 1) using a threshold of 0.5
predictions = (predicted_probabilities >= 0.5).astype(int)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)
#accuracy Accuracy: 0.9779323578795874

# Get the coefficient magnitudes and signs
coefficients = pd.DataFrame({"Feature":X_train.columns,"Coefficients":np.transpose(ridge.coef_)})

# Add a new column for the absolute value of coefficients
coefficients['abs_coefficients'] = coefficients['Coefficients'].abs()

# Sort the dataframe by the absolute value of coefficients, in descending order
coefficients = coefficients.sort_values('abs_coefficients', ascending=False)

# Display the sorted dataframe
print(coefficients)
coefficients.to_csv('coefficients_sdg16_ridge.csv', index=False)

#Visualize

# Select the top 10 features
top_10_coefficients = coefficients.head(15)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a horizontal bar plot of the coefficients
top_10_coefficients.plot(kind='barh', x='Feature', y='abs_coefficients', ax=ax)

# Set the title of the plot
ax.set_title('Feature Importance')

# Invert the y-axis to have the highest positive coefficient at the top
ax.invert_yaxis()

# Show the plot
plt.show()



