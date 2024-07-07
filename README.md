# Image Classification using CNN

This project demonstrates image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The model is trained to classify images into one of 10 classes.

## Features

- **Load Data**: Load and preprocess CIFAR-10 image data.
- **Train Model**: Train a CNN model on the CIFAR-10 training dataset.
- **Evaluate Model**: Evaluate the trained model on the CIFAR-10 test dataset.
- **Jupyter Notebook**: Interactive notebook for visualizing the image classification process.

## Download Data

Download the CIFAR-10 dataset from the official website and extract it into the data folder. Alternatively, you can run the following script to automatically download and extract the dataset:
```bash
mkdir -p data
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
cd ..
```

## Setup

1. Clone the repository and install dependencies:
  ```bash
  git clone https://github.com/SreeEswaran/Image-classification-using-CNN.git
  cd Image-classification-using-CNN
  ```

2. Install the depedencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model
   ```bash
   python scripts/train.py
   ```
2. Evaluate the model
   ```bash
   python scripts/evaluate.py
   ```

## Interactive notebook
Open the Jupyter notebook:
```bash
jupyter notebook notebooks/Image_classification_using_CNN.ipynb
```
