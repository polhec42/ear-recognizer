import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from prepare_dataset import get_train_data, get_test_data
import cv2
import math
from evaluate import evaluate_n


train_data = get_train_data()

train_dataiter = iter(train_data)

input_size = 80*40
hidden_sizes = [3000, 2000, 6000, 500, 250]
output_size = 100
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5),
    nn.Flatten(),
    nn.Linear(476*4, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], hidden_sizes[4]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[4], output_size),
    nn.Tanh(),
    nn.LogSoftmax(dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.NLLLoss()

batch_objects = next(train_dataiter)
images = batch_objects['image']
labels = batch_objects['class']

imgs = images.view(images.shape[0], -1)

logps = model(images.cuda())
 
loss = criterion(logps, labels.cuda()) # .view(-1,1)

print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print('Initial weights - ', model[0].weight)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images.cuda())
loss = criterion(output, labels.cuda())
loss.backward()
print('Gradient -', model[0].weight.grad)

# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)

#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 200

n = 10
last_n_loses = [0]*n

for e in range(epochs):
    running_loss = 0
    for batch_objects in train_data:

        images = batch_objects['image']
        labels = batch_objects['class']

        # Flatten MNIST images into a 784 long vector
        #images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    
    print("Epoch {} - Training loss: {}\n".format(e, running_loss/len(train_data)))

    if (e > n and last_n_loses[e%n] - (running_loss/len(train_data)) < 0.01) or math.isnan(running_loss):
        break
    last_n_loses[e%n] = running_loss/len(train_data)

    if e > 10:
        model.eval()
        print(evaluate_n(5, get_test_data(), model))
        model.train()
        
print("Training Time (in minutes) =",(time()-time0)/60)
torch.save(model, './my_ear_model.pt')

