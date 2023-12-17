from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim import Adam
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print(torch.cuda.is_available())

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = sorted(os.listdir(root_dir))
        self.label_mapping = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_path = os.path.join(self.root_dir, label)
        samples = [os.path.join(label_path, sample) for sample in os.listdir(label_path)]
        data = [np.load(sample) for sample in samples]

        # Use the label mapping
        label = self.label_mapping[label]

        return {'data': data, 'label': label}


class MyCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten before fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = 5  # Adjust this based on the number of classes in your dataset
model = MyCNNModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_data = CustomDataset(root_dir='dataset_preprocessed/Dataset/train')
test_data = CustomDataset(root_dir='dataset_preprocessed/Dataset/test')


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        data, label = batch['data'], batch['label']

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, label = batch['data'], batch['label']
            # Preprocess data and labels as needed
            # ...

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            accuracy = torch.sum(predicted == label).item() / len(label)

    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}')