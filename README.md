# Machine-Learning-in-Crack-Detection

**Overview**
This project implements a semantic segmentation model to detect cracks in concrete surfaces using a U-Net architecture.
It uses a custom dataset of concrete images with corresponding binary masks, where cracks are labeled as white pixels and non-crack regions as black pixels.
The model is trained to predict pixel-wise crack regions, helping in structural health monitoring and automated inspection.
🗂️ Dataset
The dataset is stored in Google Drive and organized as:
CrackConcreteDataset/
│── Train/
│   ├── images/
│   ├── masks/
│── Test/
│   ├── images/
│   ├── masks/
images/ → Raw concrete surface images
masks/ → Binary masks (1 = crack, 0 = non-crack)
⚙️ Environment Setup
This project was built and trained in Google Colab with:
Python 3.7+
TensorFlow 2.x
Keras
OpenCV
Matplotlib
NumPy
To mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')
🏗️ Model Architecture
The model follows the U-Net design:
Encoder (Downsampling path): Repeated Conv2D + MaxPooling layers
Bottleneck: Deepest feature representation
Decoder (Upsampling path): UpSampling + skip connections for spatial recovery
Output Layer: 1-channel sigmoid activation for binary mask prediction
🚀 Training
Image Size: 128 × 128
Batch Size: 8
Epochs: 25
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Training is performed using a custom DataGen generator for loading batches of images and masks.
Example:
hist = model.fit(
    train_gen,
    validation_data=valid_gen,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    epochs=25
)

**Results**
Validation Accuracy: ~95%
Validation Loss: ~0.08
Best performance around Epoch 17-20
Training & validation curves:
Accuracy Curve
Loss Curve

**Predictions**
Example predictions comparing ground truth vs predicted mask:
Left: Ground Truth Mask
Right: Predicted Mask
Overlays: Crack regions highlighted in blue

**Testing**
Testing is done using unseen images from the /Test set:
model.evaluate(test_gen, steps=test_steps)
Sample visualization includes:
Original Image
Ground Truth Mask
Predicted Mask
Overlayed Crack Segmentation

**Performance**
Model Size: ~1.9M parameters
High RAM runtime required (27GB runtime used in Colab)
GPU recommended for faster training

**Future Work**
Improve accuracy with data augmentation (rotation, flipping, brightness scaling)
Use Dice Coefficient or IoU as additional evaluation metrics
Try advanced architectures (Attention U-Net, ResU-Net, DeepLabV3+)
Deploy model via Flask API or TensorFlow Lite for real-world use

**Acknowledgments**
Dataset prepared for crack detection research
Model based on the original U-Net (Ronneberger et al., 2015)
