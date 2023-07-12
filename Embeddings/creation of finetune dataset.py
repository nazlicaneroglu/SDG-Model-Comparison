import pandas as pd

# Assuming your original dataset is named 'original_dataset.csv'
df = pd.read_csv('tokenized_data_without_punctuation_stopwords.csv')


# Filter rows based on conditions: labels_positive > labels_negative and agreement > 0.75
filtered_df = df[(df['labels_positive'] > df['labels_negative']) & (df['agreement'] > 0.75)]

# Create an empty DataFrame to store the new dataset
new_dataset = pd.DataFrame()

# Iterate over each SDG label
for sdg_label in range(1, 17):
    # Filter rows for the current SDG label and sample 100 observations
    sdg_filtered = filtered_df[filtered_df['sdg'] == sdg_label].sample(n=200, random_state=42)
    
    # Concatenate the sampled rows to the new dataset
    new_dataset = pd.concat([new_dataset, sdg_filtered], ignore_index=True)

# Save the new dataset to a CSV file
new_dataset.to_csv('new_dataset_for_ft.csv', index=False)