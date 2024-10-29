import argparse
import os
import random
from pprint import pprint
import nltk
import re
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE, Vocabulary
from nltk.lm.models import InterpolatedLanguageModel, StupidBackoff, KneserNeyInterpolated, Lidstone, WittenBellInterpolated
from nltk.util import bigrams, trigrams, ngrams
from nltk import tokenize

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
    # Tokenize sentences, convert to lowercase, and remove newlines
    text = tokenize.sent_tokenize(text.lower())
    text = [re.sub(r'\n', ' ', sent) for sent in text]  # Remove newlines
    text = [re.sub(r'[-]+', ' ', sent) for sent in text]  # Replace hyphens with spaces
    text = [re.sub(r'[^A-Za-z0-9\s-]', '', sent) for sent in text]  # Remove other non-alphanumeric characters
    text = [re.sub(r'\s+', ' ', sent).strip() for sent in text]  # Remove extra spaces and strip leading/trailing spaces
    text = [sent.split() for sent in text]
    text = [lst for lst in text if lst]  # Remove empty lists
    return text

# Apply preprocessing to all texts
texts = {file_name: preprocess_text(text) for (file_name, text) in texts.items()}

# Split training and development datasets
test_data, train_data = {}, {}
if not args.test:
    def split_train_dataset(data, ratio=0.1):
        len_data = len(data)
        n1 = int(len_data * ratio / (ratio + 1))
        indices = random.sample(range(len_data), len_data)
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

# Function to train language models
def train_language_model(model_name, n, file_name, train_text, SB_parameter=0.7, KNI_discount=0.4, Lidstone_parameter=0.5):
    ngram, vocab = padded_everygram_pipeline(n, train_text[file_name])
    model_name = model_name.upper()

    if model_name == "MLE":
        model = MLE(n)
    elif model_name == "SB":
        model = StupidBackoff(order=n, alpha=0.9)
    elif model_name == "KNI":
        model = KneserNeyInterpolated(order=n, discount=0.75)
    elif model_name == "LIDSTONE":
        model = Lidstone(order=n, gamma=0.1)
    elif model_name == "WBI":
        model = WittenBellInterpolated(order=n)
    else:
        raise Exception("Wrong model name")

    model.fit(ngram, vocab)
    return model

# Set the n-gram size
ngram_n = 2

# Train language models
print("Training language models... (this may take a while)")
language_models = {}
for file_name in train_data.keys():
    language_models[file_name] = train_language_model("LIDSTONE", ngram_n, file_name, train_data)

# Function to perform validation with backoff
def validate(train_data, test_data):
    for file_name in test_data.keys():
        correct, total = 0, 0

        for sent in test_data[file_name]:
            test_ngrams = list(ngrams(pad_both_ends(sent, n=ngram_n), n=ngram_n))

            perplexities = {}
            for train_file_name in train_data.keys():
                perplexities[train_file_name] = language_models[train_file_name].perplexity(test_ngrams)

            # Backoff to bigram if perplexity is too high
            if "MLE" in perplexities and perplexities["MLE"] > threshold:
                backoff_model = train_language_model("MLE", ngram_n, file_name, train_data)
                perplexities["MLE"] = backoff_model.perplexity(test_ngrams)

            pred_file_name = min(perplexities, key=perplexities.get)
            pred_author = extract_author_name(pred_file_name)
            actual_author = extract_author_name(file_name)

            if pred_author == actual_author:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"{extract_author_name(file_name).replace('_utf8', '')}\t{accuracy*100:.1f}% correct")

# Set the perplexity threshold for backoff
threshold = 100  # Adjust this value as needed

# Function to classify the author of a test file
def classify_test_file(train_data, test_files):
    for test_file in test_files:
        with open(test_file, 'r') as file:
            test_text = file.read()
            sentences = test_text.split('\n')

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                sentence = preprocess_text(sentence)

                test_ngrams = list(ngrams(pad_both_ends(sentence[i], n=ngram_n), n=ngram_n))
                perplexities = {}
                for file_name in train_data.keys():
                    perplexities[file_name] = language_models[file_name].perplexity(test_ngrams)

                # Backoff to bigram if perplexity is too high
                if "MLE" in perplexities and perplexities["MLE"] > threshold:
                    backoff_model = train_language_model("MLE", ngram_n, file_name, train_data)
                    perplexities["MLE"] = backoff_model.perplexity(test_ngrams)

                pred_file_name = min(perplexities, key=perplexities.get)
                pred_author = extract_author_name(pred_file_name)
                print(f"{pred_author.replace('_utf8', '')}")

# Function to generate sentences and calculate perplexity
def generate_sentences_with_perplexity(num_sents, sent_length):
    for file_name in file_names:
        print(f"Author: {file_name.replace('_utf8.txt', '')}\n")
        for i in range(num_sents):
            generated_sent = language_models[file_name].generate(sent_length, random_seed=i)
            sent_ngrams = list(ngrams(pad_both_ends(generated_sent, n=ngram_n), n=ngram_n))
            perplexities = {file_name2: round(language_models[file_name2].perplexity(sent_ngrams), 2) for file_name2 in file_names}
            print(f"Generated Sentence: {generated_sent}")
            print("Perplexity Scores:")
            for author, perplexity in perplexities.items():
                print(f"{author.replace('_utf8', '')}: {perplexity:.2f}")
            print()

# Choose whether to classify, validate, or generate sentences with perplexity
generate = False

if not generate:
    if args.test:
        test_files = [f"test_sentence_{i+1}.txt" for i in range(len(sentences))]
        classify_test_file(train_data, test_files)
    else:
        validate(train_data, test_data)
else:
    generate_sentences_with_perplexity(5, 5)
