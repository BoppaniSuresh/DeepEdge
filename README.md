# Pixel Coordinate Prediction using Deep Learning

Objective: This repository contains code to predict the coordinates (x, y) of a pixel with a value of 255 in a given 50x50 grayscale image using deep learning techniques.

### To Install requirements, run the following command:
```bash
    pip install -r /path/to/requirements.txt
```
# Dataset Generation 

Dataset.py contains code to generate random dataset for the task. The dataset is generated using a Gaussian distribution. Using a Gaussian distribution allows for the randomization of pixel coordinates, ensuring diversity in the dataset. This randomness helps the model learn to predict coordinates effectively by exposing it to various scenarios during training.

# Model Architecture 

Model.py contains the Pytorch Code for model architecture. The model architecture consists of convolutional layers. Convolutional layers are better suited for this task compared to a normal Multi-Layer Perceptron (MLP) because they can effectively capture spatial patterns in images. By analyzing local patterns and features, convolutional layers can learn to identify the pixel with a value of 255 and predict its coordinates accurately.

# Running the Code

To train the model, run the following command:
```bash
python3 train.py
```
After training, use the best epoch's path and paste it in test.py for testing the model. Then run: ``` python3 test.py ``` for testing the model

# Hyperparameters
- **Criterion**: SmoothL1Loss
  - SmoothL1Loss is chosen as the loss function for training the model. It is robust to outliers and provides a smooth gradient, making it suitable for regression tasks like predicting pixel coordinates.

- **Optimizer**: Adam
  - Adam optimizer is employed to optimize the model parameters during training. It adapts the learning rate for each parameter, resulting in faster convergence and improved performance.

- **Learning Rate**: 0.001
  - The learning rate determines the step size taken during optimization. 


# Performance Metric: R^2 Score

The performance metric used for evaluation is the R^2 score. R^2 score measures how well the model predicts the pixel coordinates compared to a simple mean prediction. It ranges from 0 to 1, where 1 indicates a perfect prediction. Using R^2 score as the performance metric is suitable for this task because it provides a clear measure of the model's predictive accuracy, helping to assess its effectiveness in predicting pixel coordinates.

# Results: 
The model achieved an R^2 score of 0.996 on the test dataset. An R^2 score of 1 indicates a perfect prediction, so a score of 0.996 indicates that the model's predictions are highly accurate and closely match the ground truth coordinates.

![image](https://github.com/BoppaniSuresh/DeepEdge/assets/118003753/c8ff4201-cbf3-4440-87ac-9c84b692d459)


