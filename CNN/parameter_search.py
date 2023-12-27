import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import optuna


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


input_size = 16
output_size = 5


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def objective(trial):
    batch_size = int(trial.suggest_categorical('batch_size', [16, 32, 64, 128]))
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    hidden_size = int(trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512, 1024]))
    num_epochs = int(trial.suggest_categorical('num_epochs', [10, 15,20, 25,30,40]))  # 固定为一个较小的数值以加快实验

    # 数据加载
    train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = CNNClassifier(input_size, hidden_size, output_size).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和评估
    accuracy = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

# 打印最佳参数
print("最佳参数: ", study.best_params)