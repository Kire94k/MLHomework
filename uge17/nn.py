import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

mytransform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST('train', download=True, train=True, transform=mytransform)
testset = datasets.MNIST('test', download=True, train=False, transform=mytransform)

#only get 20% of MNIST:

trainset = torch.utils.data.Subset(trainset, indices=range(int(0.2 * len(trainset))))
testset = torch.utils.data.Subset(testset, indices=range(int(0.2 * len(testset))))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(train_loader)
images, labels = next(dataiter)

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

lossfn = nn.CrossEntropyLoss()
lr=.01
optimizer = torch.optim.SGD(model.parameters(), lr)
iterations=10
losses = []
for epoch in range(iterations):
    running_loss = 0
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = lossfn(output, labels)
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()
print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_loader)))
losses.append(running_loss/len(train_loader))

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

testiter = iter(test_loader)
testimgs, testlabels = next(testiter)

with torch.no_grad():
    output = torch.exp(model(testimgs[0].view(1, 784))).numpy()[0]

probabilities = output / np.sum(output)
prediction = list(probabilities).index(max(probabilities))

print(np.array_str(output, precision=2, suppress_small=True))
print(np.array_str(probabilities, precision=2, suppress_small=True))

print("\nI predict:", prediction)