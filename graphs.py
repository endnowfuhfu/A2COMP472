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

random_baseline_accuracy = read_random_baseline_accuracy()

human_gold_standard_accuracy = 0.8828

# Add baseline and gold-standard to the models and accuracies lists
models.extend(['Random Baseline', 'Human Gold-Standard'])
accuracies.extend([random_baseline_accuracy, human_gold_standard_accuracy])

# Generate the bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'blue', 'blue', 'blue', 'red', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim([0, 1])

# Rotate the x-axis labels to prevent overlap
plt.xticks(rotation=45, ha='right')  # Rotate labels and adjust alignment

# Add the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

# Adjust layout to make room for the rotated x-axis labels
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
