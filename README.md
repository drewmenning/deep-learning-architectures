# deep-learning-architectures

This repository contains my submission for **M11: Exploring Deep Learning Architectures**.

## Repository Contents
- deep-learning-architectures.ipynb = report/results notebook
- deep-learning-architectures-training.ipynb = actual training and evaluation notebook
- `data/` — local dataset files used by the notebook
- `results/` — exported plots, confusion matrix, and sample predictions
- `requirements.txt` — Python packages used in this project

## Models and Results

### Part A: Multilayer Perceptron (MLP)
- Dataset: Fashion-MNIST CSV subset
- Architecture: 784 → 256 → 128 → 10
- Test accuracy: **0.809**
- Test loss: **0.571**

### Part B: Convolutional Neural Network (CNN)
- Dataset: Fashion-MNIST CSV subset
- Architecture: 2 convolution blocks + pooling + fully connected classifier
- Test accuracy: **0.837**
- Test loss: **0.529**
- Extra outputs: confusion matrix, correct predictions, incorrect predictions

### Part C: Recurrent Neural Network (RNN / LSTM)
- Dataset: Tiny Shakespeare (character-level next-character prediction)
- Model: Embedding + LSTM + linear output layer
- Final validation loss: **1.920**
- Validation perplexity: **6.82**

## Reflection Summary
The CNN was the most effective architecture for image classification in this project because it achieved the highest test accuracy on the Fashion-MNIST subset and preserved spatial structure better than the MLP. The LSTM was the right architecture for sequential text because it learned short character patterns and produced partially Shakespeare-like output after only a few epochs.

The main challenge was that the local Fashion-MNIST files available in this environment were much smaller than the full official dataset. That made the results noisier than a full benchmark run. The RNN also trained more slowly than the image models because sequence models process ordered inputs step by step.

In real-world use, MLPs fit tabular baselines, CNNs fit image inspection and classification, and RNN/LSTM models fit sequence forecasting, text generation, and sentiment-style sequence tasks.

## Notes / Assumptions
1. The available Fashion-MNIST CSV files contain **1,000 training samples** and **1,000 test samples** rather than the full official dataset size.
2. The RNN experiment uses the first **250,000 characters** from Tiny Shakespeare to keep notebook runtime manageable on CPU.
3. Part D (DCGAN) is optional bonus work and is not included in this submission.
