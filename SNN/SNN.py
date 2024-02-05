import torch
from spikingjelly.activation_based import neuron, encoding, functional
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
train_data= np.load("SNN/trainset_normalized.npz")
train_data_tensor = torch.from_numpy(train_data['data'])
train_data_tensor = train_data_tensor.permute(0,2,1) 
print(train_data_tensor.shape)
train_label_tensor = torch.from_numpy(train_data['labels'])
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)

print(train_dataset)
      
test_data= np.load("SNN/testset_normalized.npz")
test_data_tensor = torch.from_numpy(test_data['data'])
print(test_data_tensor.shape)
test_data_tensor = test_data_tensor.permute(0,2,1) 
test_label_tensor = torch.from_numpy(test_data['labels'])
test_dataset = TensorDataset(test_data_tensor, test_label_tensor)

print(test_dataset)
# data.shape = [samples, 190, 16], labels.shape = [samples]

encoder = encoding.PoissonEncoder()
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(16*180, 128)  # 输入层到隐藏层
        self.lif1 = neuron.LIFNode()  # LIF 神经元
        self.fc2 = nn.Linear(128, 5)  # 隐藏层到输出层，假设有5个类别

    def forward(self, x):
        x = x.reshape(x.size(0), -1).float()  # 将输入展平
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        return x
import tqdm
# 初始化模型和优化器
model = SNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Create a DataLoader
batch_size = 5  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in tqdm.trange(30):
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # 编码数据
        out_fr = 0.
        for t in range(30):
            encoded_img = encoder(data)
            out_fr += model(encoded_img)
        output = out_fr / 30
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        functional.reset_net(model)
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    
model.eval()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        out_fr = 0.
        for t in range(10):
            encoded_img = encoder(data)
            out_fr += model(encoded_img)
        output = out_fr / 10
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the test images: {100 * correct / total}%')
