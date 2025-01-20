import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.optim as optim
import wandb
import time

# Initialize WandB
wandb.login(key="7a42f12b660e56058d2d911cc0036220b7629317")
wandb.init(project="DeepLearningProject-CIFAR10-reg", config={
    "learning_rate": 0.1,
    "epochs": 50,
    "batch_size": 256,
    "weight_decay": 1e-4
})

config = wandb.config

# Step 1: Load and preprocess CIFAR-10 dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Step 2: Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# Model Parameters
input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 with 3 color channels
num_classes = 10          # Multiclass classification
model = LogisticRegressionModel(input_size, num_classes)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9,lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# WandB Watch
wandb.watch(model, log="all")

# Step 4: Train the Model
num_epochs = config.epochs
device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        # Flatten images
        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # Calculate training accuracy
    train_accuracy = correct_train / total_train
    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%')

    # Log training metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss,
        "train_accuracy": train_accuracy
    })

end_time = time.time()
print(f"time to train the mode: {end_time - start_time} seconds")

# Step 5: Evaluate the Model
model.eval()
y_true = []
y_pred = []
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f'Multiclass Accuracy: {accuracy * 100:.2f}%')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Log test metrics to WandB
wandb.log({
    "test_loss": test_loss / len(test_loader),
    "test_accuracy": accuracy,
    "precision": precision,
    "recall": recall
})
