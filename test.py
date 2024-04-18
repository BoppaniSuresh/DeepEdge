import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from Dataset import Dataset
from Model import MyModel

test_samples = int(0.2 * 10000)
test_dataset = Dataset(n_samples=test_samples, mean=25, std=5)
test_loader = DataLoader(test_dataset, batch_size=32)
# Load the best epoch model path
best_model = MyModel()
best_model.load_state_dict(torch.load('/home/suresh/Desktop/DeepEdge/paths/model_9_path.pth'))
best_model.eval()

predictions = []
ground_truths = []

with torch.no_grad():
    for images, targets in test_loader:
        outputs = best_model(images)
        predictions.extend(outputs.numpy())
        ground_truths.extend(targets.numpy())

predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(ground_truths[:, 0], ground_truths[:, 1], c='b', label='Ground Truth')
plt.title('Ground Truth')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(predictions[:, 0], predictions[:, 1], c='r', label='Predictions')
plt.title('Predictions')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()

plt.show()

r2 = r2_score(ground_truths, predictions)

print(f"R^2 score: {r2}")
