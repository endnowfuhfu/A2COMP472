import matplotlib.pyplot as plt

# Model names
models = [
    'word2vec_size100_window5',
    'word2vec_size200_window5',
    'word2vec_size100_window10',
    'word2vec_size200_window10'
]

# Corresponding accuracies from the analysis.csv
accuracies = [
    0.2028985507246377,
    0.2753623188405797,
    0.21739130434782608,
    0.3188405797101449
]

def read_random_baseline_accuracy(file_path='random_baseline_accuracy.txt'):
    with open(file_path, 'r') as f:
        return float(f.read())

# Use this function to get the random baseline accuracy
random_baseline_accuracy = read_random_baseline_accuracy()

human_gold_standard_accuracy = 0.8828

# Add baseline and gold-standard to the models and accuracies lists
models.extend(['Random Baseline', 'Human Gold-Standard'])
accuracies.extend([random_baseline_accuracy, human_gold_standard_accuracy])

# Generate the bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'blue', 'blue', 'blue', 'red', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim([0, 1])

# Add the accuracy values on top of the bars
for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 0.02, f'{accuracy:.2f}', ha='center')

# Save the graph
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
