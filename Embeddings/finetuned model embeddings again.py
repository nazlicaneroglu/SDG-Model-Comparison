
import pandas as pd
import torch

# Load a trained model and vocabulary that you have fine-tuned
from transformers import AutoModel, AutoTokenizer

output_dir = "C:/Users/pc/Documents/GitHub/NLP-Project/model_save"  # Replace with the actual path to your output directory

# Load the trained model
model = AutoModel.from_pretrained(output_dir)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_dir)


# Load the dataset into a pandas dataframe.
df = pd.read_csv('C:/Users/pc/Documents/GitHub/NLP-Project/new_dataset_for_ft.csv', delimiter='\t', header=None)
# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Initialize a list to store the last hidden states
last_hidden_states = []
batch_size = 10  # Adjust batch size based on memory availability

# Process sentences in batches
for i in range(0, len(df), batch_size):
    batch_sentences = df[0][i:i+batch_size]
    
    # Tokenize the batch of sentences and map tokens to their word IDs
    encoded_dict = tokenizer.batch_encode_plus(
        batch_sentences.tolist(),
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move the input tensors to the GPU if available
    encoded_dict = {key: value.to(device) for key, value in encoded_dict.items()}
    
    # Forward pass, calculate the last hidden states
    with torch.no_grad():
        result = model(**encoded_dict, return_dict=True)
    
    # Extract the last hidden states tensors
    last_hidden_states_batch = result.last_hidden_state.squeeze(dim=0)
    
    # Move the last hidden states tensors to the CPU
    last_hidden_states_batch = last_hidden_states_batch.cpu()
    
    # Append the last hidden states to the list
    last_hidden_states.append(last_hidden_states_batch)

# Concatenate the last hidden states into a single tensor
last_hidden_states = torch.cat(last_hidden_states, dim=0)

# Save the last hidden states tensor
torch.save(last_hidden_states, 'last_hidden_states.pt')
