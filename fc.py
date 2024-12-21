import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import precision_score, recall_score

wandb.login(key="7a42f12b660e56058d2d911cc0036220b7629317")
wandb.init(project="DeepLearningProject-CIFAR10-fc", config={
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 32
})

config = wandb.config

#----------------------------------------------------------------------------------------------------------------------
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        # self.fc7 = nn.Linear(32 * 32 * 3, 2048)
        self.fc6 = nn.Linear(32*32*3, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x=self.relu(self.fc7(x))
        x= self.relu(self.fc6(x))
        x= self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


#----------------------------------------------------------------------------------------------------------------------
#functions for help

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
#-----------------------------------------------------------------------------------------------------------------------
#now we load the data and normalized oit
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('Training on CPU')
else:
    print('Training on GPU')

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# Data transform to convert data to a tensor and apply normalization

# augment train and validation dataset with RandomHorizontalFlip and RandomRotation
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

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"The classes we predict are: {classes}")

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
# print(f"images shape is:{images.shape}")
#-----------------------------------------------------------------------------------------------------------------------
#create the model and specify the loss function and the optimizer
model = Net()
print(f"out model looks like this \n:{model}")

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#----------------------------------------------------------------------------------------------------------------------
#train the model
# Training the model
print("Now we train the model:")

# Number of epochs to train the model
n_epochs = 50
valid_loss_min = np.inf  # Track change in validation loss

for epoch in range(1, n_epochs + 1):
    # Keep track of training and validation loss and accuracy
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    total_train = 0

    model.train()
    for data, target in train_loader:
        # Move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # Calculate the batch loss
        loss = criterion(output, target)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()
        # Update training loss
        train_loss += loss.item() * data.size(0)
        # Calculate accuracy
        _, preds = torch.max(output, 1)
        correct_train += (preds == target).sum().item()
        total_train += target.size(0)

    # Validate the model
    model.eval()
    correct_valid = 0
    total_valid = 0
    for data, target in valid_loader:
        # Move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # Forward pass
        output = model(data)
        # Calculate validation loss
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        # Calculate validation accuracy
        _, preds = torch.max(output, 1)
        correct_valid += (preds == target).sum().item()
        total_valid += target.size(0)

    # Calculate average losses and accuracies
    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)
    train_accuracy = correct_train / total_train
    valid_accuracy = correct_valid / total_valid

    # Print training/validation statistics
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \t'
          f'Training Accuracy: {train_accuracy:.4f} \tValidation Accuracy: {valid_accuracy:.4f}')

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy
    })

    # Save the model if validation loss decreases
    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

    scheduler.step()

# Load the model with the lowest loss
model.load_state_dict(torch.load('model_cifar.pt'))

# ----------------------------------------------------------------------------------------------------------------------
# Test the model
print("Now we test our model:")
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
y_true = []
y_pred = []

# Iterate over test data
for data, target in test_loader:
    # Move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # Forward pass
    output = model(data)
    # Calculate the batch loss
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    # Convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # Compare predictions to true label
    y_true.extend(target.cpu().numpy())
    y_pred.extend(pred.cpu().numpy())
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    # Calculate test accuracy for each class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Average test loss
test_loss /= len(test_loader.dataset)
test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f'Test Loss: {test_loss:.6f}')
print(f'Test Accuracy (Overall): {test_accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Log test metrics to WandB
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "precision": precision,
    "recall": recall
})

torch.save(model.state_dict(), "pytorch_pretrained_model.pth")
wandb.save("pytorch_pretrained_model.pth")
wandb.finish()