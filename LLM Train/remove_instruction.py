from transformers import Datasets 
import pandas as pd  # Optional, for data exploration (if needed)
import numpy as np  # For cosine similarity calculations
from transformers import Datasets  # Assuming you're using Hugging Face Datasets

# Download the dataset (replace with your download method from previous discussions)
dataset = Datasets.load('open_orca/openorca')  # Assuming OpenOrca dataset

# Access training data (adjust for your specific split)
train_data = dataset['train']

def token(text):
  """Calculates the number of tokens (words in this case)."""
  return len(text.split())

def filter_function(example):
  """Filters instructions with token count >= 100."""
  return token(example['response']) >= 100  # Assuming 'response' is the instruction key

def cosine_similarity(text1, text2):
  """Calculates cosine similarity between two text vectors."""
  tokens1 = text1.split()
  tokens2 = text2.split()

  vec1 = np.zeros(len(set(tokens1 + tokens2)))
  for token in tokens1:
    vec1[tokens1.index(token)] += 1

  vec2 = np.zeros(len(set(tokens1 + tokens2)))
  for token in tokens2:
    vec2[tokens2.index(token)] += 1

  dot_product = np.dot(vec1, vec2)
  magnitude1 = np.linalg.norm(vec1)
  magnitude2 = np.linalg.norm(vec2)
  if magnitude1 > 0 and magnitude2 > 0:
    cosine_sim = dot_product / (magnitude1 * magnitude2)
  else:
    cosine_sim = 0.0
  return cosine_sim

def deduplicate_dataset(dataset, threshold=0.95):
  """Filters the dataset by removing duplicates based on cosine similarity."""
  seen_instructions = set()
  filtered_dataset = []
  for example in dataset:
    text = example['response']
    if text not in seen_instructions:
      filtered_dataset.append(example)
      seen_instructions.add(text)
    else:
      # Check for similar instructions
      for seen_text in seen_instructions:
        similarity = cosine_similarity(text, seen_text)
        if similarity >= threshold:
          break  # Skip similar instruction
  return filtered_dataset

# Filter by token count
filtered_by_token_count = train_data.filter(filter_function)

# Deduplicate based on cosine similarity
deduplicated_dataset = deduplicate_dataset(filtered_by_token_count, threshold=0.95)

# Explore the deduplicated dataset (optional)
print("Sample of deduplicated dataset:")
print(deduplicated_dataset[:5])  # Print the first 5 elements

# You can access specific elements or use the deduplicated_dataset for further processing










import matplotlib.pyplot as plt

# Download and Access Data (Replace):
# Replace the following lines with your dataset download method (Hugging Face Hub or `datasets` library) and data access code
dataset = Datasets.load('open_orca/openorca')  # Assuming OpenOrca dataset
train_data = dataset['train']

# Tokenization Function:
def tokenize_instruction(instruction):
  """Tokenizes an instruction into a list of words."""
  tokens = instruction.split()
  return tokens

# Calculate Token Counts:
token_counts = []
for example in train_data:
  text = example['response']  # Assuming 'response' is the instruction key
  token_counts.append(len(tokenize_instruction(text)))

# Plotting with Matplotlib:

# Adjust the number of bins for the histogram (optional)
bins = 20  # Adjust this parameter based on your data distribution

plt.hist(token_counts, bins=bins)
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.title("Histogram of Token Distribution")
plt.grid(True)  # Add gridlines for better readability (optional)

plt.show()