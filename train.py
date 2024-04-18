import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import Dataset
from Model import MyModel
import os 


total_samples = 10000
train_samples = int(0.6 * total_samples)
val_samples = int(0.2 * total_samples)
test_samples = total_samples - train_samples - val_samples

train_dataset = Dataset(n_samples=train_samples, mean=25, std=5)
val_dataset = Dataset(n_samples=val_samples, mean=25, std=5)
test_dataset = Dataset(n_samples=test_samples, mean=25, std=5)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model = MyModel()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_val_loss = float('inf')
best_epoch = -1
best_model_path = ''
save_dir = "/home/suresh/Desktop/DeepEdge/paths/"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    train_loss = running_train_loss / len(train_loader)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item()
    val_loss = running_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

   
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_path = os.path.join(save_dir, f"model_{best_epoch + 1}_path.pth")
        torch.save(model.state_dict(), best_model_path)

print(f"Best epoch: {best_epoch + 1}, Best val loss: {best_val_loss}")



