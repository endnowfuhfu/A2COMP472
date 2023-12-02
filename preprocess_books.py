import os
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

def clean_text(text):
    """
    Function to clean text by removing punctuation and converting to lowercase.
    """
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

def preprocess_book(file_path):
    """
    Function to preprocess a single book.
    Returns a list of sentences, where each sentence is a list of words.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = clean_text(text)
        sentences = sent_tokenize(text)
        word_lists = [word_tokenize(sentence) for sentence in sentences]
        return word_lists

def preprocess_books(folder_path):
    """
    Preprocess all books in the specified folder.
    Returns a list of sentences from all books, where each sentence is a list of words.
    """
    all_sentences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            book_sentences = preprocess_book(file_path)
            all_sentences.extend(book_sentences)
    return all_sentences

