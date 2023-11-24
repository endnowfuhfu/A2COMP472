import gensim.downloader as api
import json
import pandas as pd
import random

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

def task1_details(model, synonym_dataset):
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
    details_file = "word2vec-google-news-300-details.csv"
    details_df.to_csv(details_file, index=False)

    return correct_labels, questions_without_guessing

def task1_analysis(model, correct_labels, questions_without_guessing):

     # Vocabulary size, correct labels, questions without guessing, and accuracy
    vocab_size = len(model.key_to_index)    

    accuracy = (correct_labels / questions_without_guessing) if questions_without_guessing > 0 else 0

    # Save analysis to csv
    analysis_data = [
        ["word2vec-google-news-300", vocab_size, correct_labels, questions_without_guessing, accuracy]
    ]

    columns = ['model name', 'size of vocabulary', 'number of correct labels', 'questions without guessing', 'accuracy']
    analysis_df = pd.DataFrame(analysis_data, columns=columns)

    analysis_file = "analysis.csv"
    analysis_df.to_csv(analysis_file, index=False)


def main():
    model = api.load("word2vec-google-news-300")

    with open('synonym.json', 'r') as file:
        synonym_dataset = json.load(file)

    correct_labels, questions_without_guessing = task1_details(model, synonym_dataset)

    task1_analysis(model, correct_labels, questions_without_guessing)
    
   

if __name__ == "__main__":
    main()