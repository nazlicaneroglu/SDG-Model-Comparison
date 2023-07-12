# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:50:00 2023

@author: pc
"""
 
import pandas as pd
import torch
import numpy as np

# Load the dataset into a pandas dataframe.
#df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df = pd.read_csv('C:/Users/pc/Documents/GitHub/NLP-Project/new_dataset_for_ft.csv', delimiter='\t', header=None)
# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))



# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# Initialize a list to store the last hidden states


# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       return_dict=True)

    logits = result.pooler_output

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits) 
    true_labels.append(label_ids)

print('DONE.')

import torch

# Assuming result is the BaseModelOutputWithPoolingAndCrossAttentions object
last_hidden_state = result.last_hidden_state

# Save the last_hidden_state tensor
torch.save(last_hidden_state, 'last_hidden_state_finetunedmodel.pt')

import numpy as np

# Assuming result is the BaseModelOutputWithPoolingAndCrossAttentions object
last_hidden_state = result.last_hidden_state.numpy()

# Save the last_hidden_state as a NumPy array file
np.save('last_hidden_state_finetunedmodel.npy', last_hidden_state)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = np.concatenate(true_labels, axis=0)

accuracy = sum(1 for (x,y) in zip(flat_predictions, flat_true_labels) if x == y) / len(flat_true_labels)
print(accuracy)

flat_predictions = np.concatenate(predictions, axis=0)

