import numpy as np
import torch
from torch.utils.data import Dataset

# My function to generate the dataset 

class Dataset(Dataset):
    def __init__(self, n_samples, mean, std):
        self.n_samples = n_samples
        self.mean = mean 
        self.std = std
    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        try:
            # Generating the random 50x50 pixel image
            image = np.zeros((50,50), dtype = np.float32)
            # Generate random coordinates following a Gaussian distribution
            x = int(np.random.normal(self.mean, self.std))
            y = int(np.random.normal(self.mean, self.std))

            # Clip coordinates to ensure they fall within the image bounds
            x = np.clip(x, 0, 49)
            y = np.clip(y, 0, 49)
            
            image[x,y] = 255
            
            #Normalizing the image 
            image = image/255.0
            # To add Batch dimension
            image = torch.tensor(image).unsqueeze(0)
            #print('shape of image after adding batch dimension is : ', image.shape) # [1,50,50]
            label = torch.tensor([x,y], dtype = torch.float32)  
            
            return image, label
        except Exception as e:
            print(f"Error occurred while generating sample at index {index}: {e}")
            return None, None
             
