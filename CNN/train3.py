import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Fully connected layers
        self.fc1 = nn.Linear(output_size * (180 // 4), 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Assuming x has shape (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)  # Permute dimensions for Conv1d
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def load_data(file_name):
    with np.load(file_name) as data:
        data_array = data['data']
        labels_array = data['labels']
    return data_array, labels_array


def to_tensor(data, labels):
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.LongTensor(labels.argmax(axis=1))
    return data_tensor, labels_tensor


train_data, train_labels = load_data('trainset_normalized.npz')
test_data, test_labels = load_data('testset_normalized.npz')

train_data_tensor, train_labels_tensor = to_tensor(train_data, train_labels)
test_data_tensor, test_labels_tensor = to_tensor(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=batch_size, shuffle=False)

# Hyperparameters
input_size = 16
hidden_size = 64
output_size = 5
num_layers = 3
learning_rate = 0.0007640405444859117
num_epochs = 100
num_models = 5

model = CNNClassifier(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

models = []
for i in range(num_models):
    model = CNNClassifier(input_size, hidden_size, output_size).to(device)
    models.append(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for model in models:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 评估模型
predictions = []
for model in models:
    model.eval()
    model_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            model_preds.extend(predicted.cpu().numpy())
    predictions.append(model_preds)


all_labels = test_labels_tensor.numpy()


ensemble_preds = []
for i in range(len(predictions[0])):

    votes = [predictions[j][i] for j in range(num_models)]
    most_common = Counter(votes).most_common(1)[0][0]
    ensemble_preds.append(most_common)

accuracy = accuracy_score(all_labels, ensemble_preds)
f1 = f1_score(all_labels, ensemble_preds, average='weighted')

print(f'Ensemble Accuracy: {accuracy:.4f}')
print(f'Ensemble F1 Score: {f1:.4f}')