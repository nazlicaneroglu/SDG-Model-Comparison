#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:09:01 2023

@author: admin
"""

import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from gensim import corpora, models
import numpy as np
torch.device('mps')
# Load a trained model and vocabulary that you have fine-tuned
from transformers import AutoModel, AutoTokenizer

output_dir = "/Users/admin/Documents/conda/model_save"  # Replace with the actual path to your output directory


# Load the vectors from the first Torch file
file1 = '/Users/admin/Documents/conda/adsız klasör/**first batch finetuned1.tf'
vectors1 = torch.load(file1)

# Load the vectors from the second Torch file
file2 = '/Users/admin/Documents/conda/adsız klasör/**last batch finetuned1.tf'
vectors2 = torch.load(file2)

# Concatenate the vectors
concatenated_vectors = torch.cat((vectors1, vectors2), dim=0)

# Save the concatenated vectors to a new Torch file
output_file = 'finetunedembeddings.torch'
torch.save(concatenated_vectors, output_file)

dataset = pd.read_csv('/Users/admin/Documents/conda/new_dataset_for_ft.csv')
print(dataset.shape)
dataset.head()
text_data=dataset["text"]


# Load the tensor file
concatenated_vectors = torch.load('finetunedembeddings.torch')

# Get the shape of the tensor
num_rows, num_cols = concatenated_vectors.shape

# Create a dictionary to store the columns
columns_dict = {}

# Iterate over each column and store it in the dictionary
for col_idx in range(num_cols):
    column_data = concatenated_vectors[:, col_idx].tolist()  # Get the column data as a list
    columns_dict[f'Column_{col_idx+1}'] = column_data  # Store the column data in the dictionary

# Create a DataFrame from the dictionary
new_dataset = pd.DataFrame(columns_dict)

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming your dataset is stored in the 'data' variable

# Step 1: Data preprocessing
# Handle missing values, normalize data, remove outliers, etc.

# Step 2: Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_dataset)

# Step 3: Compute covariance matrix
cov_matrix = np.cov(new_dataset.T) #i have changed this part to see how non scaling works

# Step 4: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Select principal components
# Sort eigenvalues in descending order
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Choose the top-k eigenvectors
k = 10  # Number of principal components
selected_components = np.array([eigen_pairs[i][1] for i in range(k)])

# Step 6: Project data onto principal components
projected_data = np.dot(scaled_data, selected_components.T)


#k-means clustering
from sklearn.cluster import KMeans
km = KMeans(
    n_clusters=16, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(new_dataset)
df = pd.DataFrame({'text' :text_data, 'topic_cluster' :y_km })
#I saved the dataframe

token_lists = dataset['word_split'].tolist()
token_lists = [d.split() for d in token_lists]
k_clustering_data = pd.read_csv('/Users/admin/Documents/conda/adsız klasör/LDA/dataset_kmeans and tokens.csv')


from wordcloud import WordCloud
all_words = ''.join([word for word in k_clustering_data['text'][0:100000]])
all_words
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Some frequent words used in all of the dataset", weight='bold', fontsize=14)
plt.show()

def get_top_n_words(corpus, n=10):
  vec = CountVectorizer(stop_words='english').fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in   vec.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq[:n]
words_all = []
word_values_all = []
num_clusters = 16
for cluster in range(num_clusters):
    subset_dataset = k_clustering_data[k_clustering_data['topic_cluster'] == cluster]
    words = []
    word_values = []
    
    for i, j in get_top_n_words(subset_dataset['text'], 15):
        words.append(i)
        word_values.append(j)
    
    words_all.append(words)
    word_values_all.append(word_values)

# Generate separate plots for each cluster
for cluster in range(num_clusters):
    words = words_all[cluster]
    word_values = word_values_all[cluster]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(words)), word_values)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation='vertical')
    ax.set_title(f'Top 15 words in Cluster {cluster}')
    ax.set_xlabel('Word')
    ax.set_ylabel('Number of occurrences')
    plt.show()
    
#k-means cluster for PCA'd data

#k-means clustering
from sklearn.cluster import KMeans
km = KMeans(
    n_clusters=16, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km_pca = km.fit_predict(projected_data)
k_clustering_data_pca_kfive=pd.DataFrame({'text' :text_data, 'topic_cluster' :y_km_pca})
#I saved the dataframe


words_all = []
word_values_all = []
num_clusters = 16
for cluster in range(num_clusters):
    subset_dataset = k_clustering_data_pca_withoutscaling[k_clustering_data_pca_withoutscaling['topic_cluster'] == cluster]
    words = []
    word_values = []
    
    for i, j in get_top_n_words(subset_dataset['text'], 15):
        words.append(i)
        word_values.append(j)
    
    words_all.append(words)
    word_values_all.append(word_values)

# Generate separate plots for each cluster
for cluster in range(num_clusters):
    words = words_all[cluster]
    word_values = word_values_all[cluster]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(words)), word_values)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation='vertical')
    ax.set_title(f'Top 15 words in Cluster (PCA){cluster}')
    ax.set_xlabel('Word')
    ax.set_ylabel('Number of occurrences')
    plt.show()
    
#getting the most occuring cluster groups
# with 'text' and 'topic_cluster' columns

segment_size = 200
total_rows = len(k_clustering_data_pca_kfive)

# Iterate over segments
for i in range(0, total_rows, segment_size):
    # Get the current segment
    segment = k_clustering_data_pca_kfive['topic_cluster'].iloc[i:i+segment_size]

    # Find the most frequent value in the segment
    most_frequent_value = segment.mode().values[0]

    print(f"Segment {i//segment_size + 1}: Most frequent value = {most_frequent_value}")