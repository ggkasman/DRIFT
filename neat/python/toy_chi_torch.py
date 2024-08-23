import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Assume the necessary utility functions and classes from your previous translations are available

# Data simulation
n = 10000
p = 1
y = np.random.chisquare(df=4, size=n).reshape(-1, 1).astype(np.float32)
X = np.random.normal(size=(n, p)).astype(np.float32)
model_type = "LS"

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, 1)
        self.fc_sd = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        sd = torch.exp(-0.5 * self.fc_sd(x))
        return mu, sd

class NEATModel(nn.Module):
    def __init__(self, input_dim):
        super(NEATModel, self).__init__()
        self.net = SimpleNet(input_dim)

    def forward(self, x, y):
        mu, sd = self.net(x)
        pred = (y - mu) * sd
        return pred

# Define model, loss, optimizer
input_dim = p
neat_model = NEATModel(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(neat_model.parameters(), lr=0.0001)

# Prepare data
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=400, shuffle=False)

# Training loop
patience = 100
best_loss = float('inf')
trigger_times = 0

for epoch in range(500):
    neat_model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = neat_model(batch_x, batch_y)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(train_loader.dataset)

    # Validation
    neat_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = neat_model(batch_x, batch_y)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        best_model_wts = neat_model.state_dict().copy()
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

neat_model.load_state_dict(best_model_wts)

# Predictions
neat_model.eval()
with torch.no_grad():
    pred_neat = neat_model(torch.tensor(X), torch.tensor(y)).numpy()

# Plotting
plt.scatter(pred_neat, y)
plt.xlabel("pred_neat")
plt.ylabel("y")
plt.show()

probplot(pred_neat.flatten(), plot=plt)
plt.show()