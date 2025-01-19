# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import wandb
import sklearn
from sklearn.metrics import precision_score, recall_score

wandb.login(key="7a42f12b660e56058d2d911cc0036220b7629317")
wandb.init(project="DeepLearningProject-CIFAR10-CNN", config={
    "learning_rate": 0.001,
    "epochs": 400,
    "batch_size": 32
})

config = wandb.config
#----------------------------------------------------------------------------------------------------------------------
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.batchNorm5 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.batchNorm6 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.batchNorm7 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 10)
        self.batchNorm8 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.maxpool(self.relu(self.batchNorm2(self.conv2(x))))
        x = self.relu(self.batchNorm3(self.conv3(x)))
        x = self.maxpool(self.relu(self.batchNorm4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(self.relu(self.batchNorm5(self.fc1(x))))
        x = self.dropout(self.relu(self.batchNorm6(self.fc2(x))))
        x = self.dropout(self.relu(self.batchNorm7(self.fc3(x))))
        # x = self.dropout(self.relu(self.batchNorm8(self.fc4(x))))

        x = self.fc4(x)
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
train_data = datasets.CIFAR10('data', train=True,download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,download=True, transform=test_transform)

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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers)

# specify the image classes
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print(f"The classes we predict are: {classes}")

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
# print(f"images shape is:{images.shape}")

# Plot the images in the batch, along with the corresponding labels
print("How the images we try to predict look like:")
fig = plt.figure(figsize=(25, 4))

# # Display 20 images
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
#     imshow(images[idx])  # Unnormalize and plot
#     ax.set_title(classes[labels[idx]])  # Add title
# plt.show()  # Display the figure
#
#
# print("And one image in more specific detail:")
# rgb_img = np.squeeze(images[3])  # Extract a single image (C, H, W)
# channels = ['Red Channel', 'Green Channel', 'Blue Channel']
#
# fig = plt.figure(figsize=(36, 36))
# for idx in np.arange(rgb_img.shape[0]):
#     ax = fig.add_subplot(1, 3, idx + 1)
#     img = rgb_img[idx]
#     ax.imshow(img, cmap='gray')  # Display individual channels in grayscale
#     ax.set_title(channels[idx])
#     plt.axis('off')
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#create the model and specify the loss function and the optimizer
model = Net()
print(f"out model looks like this \n:{model}")

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function- using cross entropy
criterion = nn.CrossEntropyLoss()

# specify optimizer-using Adam
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)

#----------------------------------------------------------------------------------------------------------------------
#train the model
print("Training the model...")
n_epochs = config.epochs
valid_loss_min = np.inf

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training phase
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, preds = torch.max(output, 1)
        correct_train += (preds == target).sum().item()
        total_train += target.size(0)

    train_accuracy = correct_train / total_train

    # Validation phase
    correct_valid = 0
    total_valid = 0
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        _, preds = torch.max(output, 1)
        correct_valid += (preds == target).sum().item()
        total_valid += target.size(0)

    valid_loss /= len(valid_loader.sampler)
    train_loss /= len(train_loader.sampler)
    valid_accuracy = correct_valid / total_valid

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy
    })

    print(f"Epoch: {epoch} \tTrain Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f} "
          f"\tTrain Accuracy: {train_accuracy:.4f} \tValidation Accuracy: {valid_accuracy:.4f}")

    # Save the best model
    if valid_loss <= valid_loss_min:
        print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

# Load the best model
model.load_state_dict(torch.load('model_cifar.pt'))

# Test the model
print("Testing the model...")
test_loss = 0.0
class_correct = [0. for _ in range(100)]
class_total = [0. for _ in range(100)]
model.eval()
y_true = []
y_pred = []

for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, preds = torch.max(output, 1)
    y_true.extend(target.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())
    correct_tensor = preds.eq(target.data.view_as(preds))
    correct = np.squeeze(correct_tensor.cpu().numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss /= len(test_loader.dataset)
test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Log test results to WandB
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "precision": precision,
    "recall": recall
})

#----------------------------------------------------------------------------------------------------------------------
# #visualize the results
#
# # obtain one batch of test images
# dataiter = iter(test_loader)
# images, labels = next(dataiter)
# images.numpy()
#
# # move model inputs to cuda, if GPU available
# if train_on_gpu:
#     images = images.cuda()
#
# # get sample outputs
# output = model(images)
# # convert output probabilities to predicted class
# _, preds_tensor = torch.max(output, 1)
# preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
#
# # Plot the images in the batch, along with predicted and true labels
# fig = plt.figure(figsize=(25, 4))
#
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
#     # Unnormalize and plot
#     imshow(images[idx] if not train_on_gpu else images[idx].cpu())
#     ax.set_title("{} ({})".format(
#         classes[preds[idx]], classes[labels[idx]]),
#         color=("green" if preds[idx] == labels[idx].item() else "red")
#     )
# plt.show()

