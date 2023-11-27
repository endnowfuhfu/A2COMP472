import gensim.downloader as api
import json
import pandas as pd
import random
import os

# Function to find the closest synonym
def find_closest_synonym(model, question_word, answer_words):
    try:
        if question_word in model:
            cosine_sims = []
    
            for word in answer_words:
                if word in model:
                    similarity = model.similarity(question_word, word)
                    cosine_sims.append((word, similarity))
    
            # Sort the similarities in descending order (from the second value in the cosine pair))
            cosine_sims.sort(key=lambda pair: pair[1], reverse=True)
            
            # Return the closest synonym
            return cosine_sims[0][0] if cosine_sims else None
        else:
            return None
    except KeyError:
        return None
    
def no_answer_words_in_model(model, answer_words):
    for word in answer_words:
        if word in model:
            return False
    return True

def task_details(model, synonym_dataset, modelname):
    output_results = []
    correct_labels = 0
    questions_without_guessing = 0

    for item in synonym_dataset:
        question_word = item['question']
        answer_words = item['choices']
        
        # Find the closest synonym
        guess_word = find_closest_synonym(model, question_word, answer_words)
        is_no_word_present = guess_word is None and no_answer_words_in_model(model, answer_words)

        # Determine the label
        if is_no_word_present or (guess_word is None or no_answer_words_in_model(model, answer_words)):
            #make the guess word a random word from the answer words
            guess_word = random.choice(answer_words)
            label = 'guess'
        elif guess_word == item['answer']:
            label = 'correct'
            correct_labels += 1
            questions_without_guessing += 1
        else:
            label = 'wrong'
            questions_without_guessing += 1
        
        output_results.append([question_word, item['answer'], guess_word, label])

    # Create a DataFrame from the output_results list
    columns = ['question-word', 'correct', 'guess', 'label']
    details_df = pd.DataFrame(output_results, columns=columns)

    # Use the DataFrame to save the results in a csv file
    details_file = modelname+"-details.csv"
    details_df.to_csv(details_file, index=False)

    return correct_labels, questions_without_guessing

def task_analysis(model, correct_labels, questions_without_guessing, modelname):

     # Vocabulary size, correct labels, questions without guessing, and accuracy
    vocab_size = len(model.key_to_index)    

    accuracy = (correct_labels / questions_without_guessing) if questions_without_guessing > 0 else 0

    # Save analysis to csv
    analysis_data = [
        [modelname, vocab_size, correct_labels, questions_without_guessing, accuracy]
    ]

    columns = ['model name', 'size of vocabulary', 'number of correct labels', 'questions without guessing', 'accuracy']
    analysis_df = pd.DataFrame(analysis_data, columns=columns)

    analysis_file = "analysis.csv"

     # Check if the file exists and if it does, do not write headers
    file_exists = os.path.isfile(analysis_file)
    
    # Open the file in append mode ('a') if it exists or write mode ('w') otherwise
    with open(analysis_file, 'a' if file_exists else 'w', newline='') as f:
        analysis_df.to_csv(f, header=not file_exists, index=False)


def main():
    model = api.load("word2vec-google-news-300")
    model2 = api.load("glove-wiki-gigaword-100")
    model3 = api.load("glove-twitter-200")
    model4 = api.load("glove-wiki-gigaword-300")
    model5 = api.load("glove-wiki-gigaword-200")
    #model6 = api.load("fasttext-wiki-news-subwords-300")
    
    with open('synonym.json', 'r') as file:
        synonym_dataset = json.load(file)

    correct_labels, questions_without_guessing = task_details(model, synonym_dataset, "word2vec-google-news-300")
    task_analysis(model, correct_labels, questions_without_guessing, "word2vec-google-news-300")

    correct_labels, questions_without_guessing = task_details(model2, synonym_dataset, "glove-wiki-gigaword-100")
    task_analysis(model2, correct_labels, questions_without_guessing, "glove-wiki-gigaword-100")

    correct_labels, questions_without_guessing = task_details(model3, synonym_dataset, "glove-twitter-200")
    task_analysis(model3, correct_labels, questions_without_guessing, "glove-twitter-200")

    correct_labels, questions_without_guessing = task_details(model4, synonym_dataset, "glove-wiki-gigaword-300")
    task_analysis(model4, correct_labels, questions_without_guessing, "glove-wiki-gigaword-300")

    correct_labels, questions_without_guessing = task_details(model5, synonym_dataset, "glove-wiki-gigaword-200")
    task_analysis(model5, correct_labels, questions_without_guessing, "glove-wiki-gigaword-200")
 

if __name__ == "__main__":
    main()