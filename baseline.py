import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torchvision
from torchvision.transforms import transforms
import wandb

# Initialize WandB
wandb.login(key="7a42f12b660e56058d2d911cc0036220b7629317")
wandb.init(project="DeepLearningProject-CIFAR10-baseline")

# Load the CIFAR-10 Test Dataset
transform = transforms.ToTensor()
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Step 2: Generate Random Predictions
np.random.seed(42)
num_classes = 10

# Generate random class predictions for each image
random_predictions = np.random.randint(0, num_classes, size=len(test_dataset))

# Step 3: Retrieve True Labels
true_labels = [label for _, label in test_dataset]

# Step 4: Calculate Metrics
accuracy = accuracy_score(true_labels, random_predictions)
precision = precision_score(true_labels, random_predictions, average='macro', zero_division=0)
recall = recall_score(true_labels, random_predictions, average='macro', zero_division=0)

print(f"Accuracy of random baseline: {accuracy * 100:.2f}%")
print(f"Precision of random baseline: {precision * 100:.2f}%")
print(f"Recall of random baseline: {recall * 100:.2f}%")


# Log metrics for multiple test runs or steps
for step in range(1, 4):  # Example: 3 test runs
    wandb.log({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })
