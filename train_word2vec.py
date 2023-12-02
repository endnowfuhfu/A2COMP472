import os
import gensim
from gensim.models import Word2Vec
import json
import csv
from gensim.models import KeyedVectors
from preprocess_books import preprocess_books
import random

def train_word2vec(sentences, vector_size, window, model_name):
    """
    Train a Word2Vec model and save it.
    """
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1, workers=4)
    model.save(f"{model_name}.model")
    print(f"Model {model_name} trained and saved.")
    return model

def load_model(model_name):
    """
    Load a Word2Vec model.
    """
    return Word2Vec.load(f"{model_name}.model")

def load_synonym_test_data(json_file_path):
    with open(json_file_path, 'r') as json_file:
        synonym_test_data = json.load(json_file)
    return synonym_test_data

def evaluate_model(model, synonym_test_data, model_name):
    correct_count = 0
    total_count = 0
    guess_count = 0
    details = []
    
    for item in synonym_test_data:
        question_word = item['question']
        answer_word = item['answer']
        guess_words = item['choices']
        most_similar_word = None
        
        # Check if the words are in the model's vocabulary
        if question_word in model.wv:
            valid_guess_words = [word for word in guess_words if word in model.wv]
            if valid_guess_words:
                most_similar_word = max(valid_guess_words, key=lambda word: model.wv.similarity(question_word, word))
                label = 'correct' if most_similar_word == answer_word else 'wrong'
                if label == 'correct':
                    correct_count += 1
            else:
                label = 'guess'
                guess_count += 1
        else:
            label = 'guess'
            guess_count += 1
        
        total_count += 1
        details.append((question_word, answer_word, most_similar_word if most_similar_word else "N/A", label))
    
    accuracy = correct_count / (total_count - guess_count) if total_count - guess_count > 0 else 0
    summary = (model_name, len(model.wv.key_to_index), correct_count, total_count - guess_count, accuracy)
    
    return details, summary

def write_details_csv(model_name, details):
    """
    Write the detailed evaluation results to a CSV file.
    """
    filename = f"{model_name}-details.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question_word', 'correct_answer', 'guess', 'label'])
        for detail in details:
            writer.writerow(detail)

def update_analysis_csv(summary):
    """
    Append the summary statistics of the model to the analysis.csv file.
    """
    filename = 'analysis.csv'
    # Check if analysis.csv exists, if not, create it and write headers
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['model_name', 'vocabulary_size', 'number_correct', 'number_answered', 'accuracy'])
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(summary)

###########################################
#             MODEL TRAINING              #
###########################################

# Path to the folder containing books and to the synonym test data
books_folder = 'books'
synonym_test_data_path = 'synonym.json'

# Preprocess the books to get the sentences
sentences = preprocess_books(books_folder)

# Load the synonym test data
synonym_test_data = load_synonym_test_data(synonym_test_data_path)

# Define model parameters
window_sizes = [5, 10]
embedding_sizes = [100, 200]

# Train, evaluate models and save results
for window in window_sizes:
    for size in embedding_sizes:
        model_name = f"word2vec_size{size}_window{window}"
        model = train_word2vec(sentences, size, window, model_name)
        
        # Evaluate the model
        details, summary = evaluate_model(model, synonym_test_data, model_name)
        
        # Write to <model_name>-details.csv
        write_details_csv(model_name, details)
        
        # Update analysis.csv
        update_analysis_csv(summary)

print("All models trained, evaluated, and results saved.")


#########################
#    RANDOM BASELINE    #
#        MODEL          #
#########################

def random_baseline_accuracy(synonym_test_data):
    correct_count = 0
    for item in synonym_test_data:
        guess_word = random.choice(item['choices'])
        if guess_word == item['answer']:
            correct_count += 1
    return correct_count / len(synonym_test_data)

# At the end of train_word2vec.py
random_accuracy = random_baseline_accuracy(synonym_test_data)
print("Random Baseline Accuracy:", random_accuracy)

# Save the random baseline accuracy to a file
with open('random_baseline_accuracy.txt', 'w') as f:
    f.write(str(random_accuracy))

