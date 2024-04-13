# Team ID:			[ GG_3303 ]
# Author List:		[ Atharva Satish Attarde, Nachiket Ganesh Apte, Ashutosh Anil Dongre, Prachit Suresh Deshinge ]
# Filename:			task_2c_model_training


"""This code is to be run on google colab only"""


from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import os
import requests
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import cv2

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "/content/drive/MyDrive/training50x50"
dataset = datasets.ImageFolder(data_dir, transform=data_transform)

from sklearn.model_selection import train_test_split
train_size = 0.8
train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, shuffle=True, random_state=42)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

vgg16 = torchvision.models.vgg16(pretrained=True)
feature_extractor = nn.Sequential(*list(vgg16.features.children())[:-1])
custom_classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4608, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 5)
)

model = nn.Sequential(
    feature_extractor,
    custom_classifier
)

for param in feature_extractor.parameters():
    param.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 100
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracy_list.append(train_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracy_list.append(accuracy)


epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_accuracy_list, label='Training Accuracy')
plt.plot(epochs, test_accuracy_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()

torch.save(model, 'vgg1650x50new.pth')


%cd /content/drive/My Drive/
!cp '/content/vgg1650x50new.pth' '/content/drive/My Drive/'