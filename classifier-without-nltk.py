import argparse
import os
import random
import numpy as np
import re
from pprint import pprint
from collections import defaultdict

# Define command-line arguments
parser = argparse.ArgumentParser(description='Author classifier')
parser.add_argument('authorlist', type=str, help='Path to author list file')
parser.add_argument('-test', type=str, help='Path to test file')
args = parser.parse_args()

# Read the author list from the command line
with open(args.authorlist, 'r') as author_list_file:
    author_list_contents = author_list_file.read()
file_names = author_list_contents.splitlines()

# Function to extract author name from a file name
def extract_author_name(file_name):
    return file_name.rsplit(".", 1)[0]

# Load the text from training files
texts = {}
for file_name in file_names:
    with open("ngram_authorship_train/" + file_name, "r") as file:
        text = file.read()
        texts[file_name] = text
if args.test:
    with open(args.test, 'r') as test_file:
        test_file_contents = test_file.read()
        sentences = test_file_contents.split('\n')

    for i, sentence in enumerate(sentences):
        # Skip empty sentences
        if sentence.strip():
            with open(f"test_sentence_{i+1}.txt", 'w') as test_sentence_file:
                test_sentence_file.write(sentence.strip())

# Preprocess text
def preprocess_text(text):
    # Lower case
    text = re.split(r'[.!?]', text.lower())
    # Remove newline
    text = [sent.replace('\n', ' ').replace('-', ' ') for sent in text]
    # Remove punctuations
    text = [re.sub(r'[^A-Za-z0-9- ]+', '', sent) for sent in text]
    # Remove extra space
    text = [' '.join(sent.split()) for sent in text]
    # Split sentence into word lists    
    text = [sent.split(' ') for sent in text]
    ## Remove empty lists
    text = [lst for lst in text if lst]
    return text

# Apply preprocessing to all texts
texts = {file_name: preprocess_text(text) for (file_name, text) in texts.items()}
#print(texts)

# Split training and development datasets
test_data, train_data = {}, {}
if not args.test:
    def split_train_dataset(data, ratio=0.1):
        n = len(data)
        n1 = int(n * ratio / (ratio + 1))
        indices = random.sample(range(n), n)
        data1 = [data[i] for i in indices[:n1]]
        data2 = [data[i] for i in indices[n1:]]
        return data1, data2

    print("Splitting into training and development datasets...")
    for (file_name, text) in texts.items():
        test_data[file_name], train_data[file_name] = split_train_dataset(text)
else:
    for (file_name, text) in texts.items():
        if file_name != args.test:
            train_data[file_name] = texts[file_name]
        else:
            test_data[file_name] = texts[file_name]
           # print(test_data)

# Function to train bigram language models
def train_bigram_model(train_text):
    bigram_model = defaultdict(lambda: defaultdict(int))
    
    for sentence in train_text:
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i], sentence[i + 1]
            bigram_model[word1][word2] += 1
    
    return bigram_model

# Set the n-gram size (bigram)
ngram_n = 2

# Train bigram language models
print("Training bigram models...")
bigram_models = {}
for file_name in train_data.keys():
    bigram_models[file_name] = train_bigram_model(train_data[file_name])

# Function to calculate perplexity for bigram models
def calculate_perplexity(test_sentence, model):
    perplexity = 1.0
    V = len(model)
    for i in range(len(test_sentence) - 1):
        word1, word2 = test_sentence[i], test_sentence[i + 1]
        bigram_count = model[word1][word2]
        unigram_count = sum(model[word1].values())
        conditional_probability = (bigram_count + 1) / (unigram_count + V)
        perplexity *= 1 / conditional_probability
    return perplexity

# Function to validate the bigram models
def validate_bigram_models(train_data, test_data, bigram_models):
    for file_name in test_data.keys():
        correct, total = 0, 0
        for sentence in test_data[file_name]:
            perplexities = {author: calculate_perplexity(sentence, bigram_models[author]) for author in train_data.keys()}
            predicted_author = min(perplexities, key=perplexities.get)
            # Remove the ".txt" extension
            predicted_author = predicted_author.replace(".txt", "")
            actual_author = extract_author_name(file_name)
        
            #print(f"Predicted Author: {predicted_author}, Actual Author: {actual_author}")
        
            # Check if the predicted author matches the actual author
            if predicted_author == actual_author:
                correct += 1
        
            total += 1
        
        accuracy = correct / total
        print(f"{extract_author_name(file_name).replace('_utf8', '')}\t{accuracy*100:.1f}% correct")



# Validate the bigram models
validate_bigram_models(train_data, test_data, bigram_models)
