{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "if tf.config.list_physical_devices('GPU'):\n",
    "    print('GPU is available.')\n",
    "else:\n",
    "    print('GPU is NOT available. Make sure TensorFlow is installed with GPU support and NVIDIA GPU drivers are installed.')\n",
    "\n",
    "# Define and configure TensorFlow session\n",
    "config = tf.compat.v1.ConfigProto()\n",
    "config.gpu_options.allow_growth = True\n",
    "session = tf.compat.v1.Session(config=config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.applications import VGG16\n",
    "from tensorflow.keras.layers import Flatten\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.layers import Input\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.preprocessing.image import img_to_array\n",
    "from tensorflow.keras.preprocessing.image import load_img\n",
    "from tensorflow.keras.models import load_model\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import numpy as np\n",
    "import mimetypes\n",
    "import argparse\n",
    "import imutils\n",
    "import os\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "MODEL_PATH = \"thermal_v1.h5\"\n",
    "model = load_model(MODEL_PATH)\n",
    "a=3\n",
    "b=2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the path to the testing image\n",
    "imagePath = r\"Y:\\_Prjs\\solar\\solarthermal\\testimgs\\5.jpg\"\n",
    "\n",
    "# Load and preprocess the testing image\n",
    "image = load_img(imagePath, target_size=(224, 224))\n",
    "image = img_to_array(image) / 255.0\n",
    "image = np.expand_dims(image, axis=0)\n",
    "\n",
    "# Make predictions using the trained model\n",
    "preds = model.predict(image)[0]\n",
    "\n",
    "# Extract predicted bounding box coordinates\n",
    "(startX, startY, endX, endY) = preds\n",
    "\n",
    "# Read the original image and resize it\n",
    "image = cv2.imread(imagePath)\n",
    "image = imutils.resize(image, width=600)\n",
    "(h, w) = image.shape[:2]\n",
    "\n",
    "# Scale the predicted coordinates to match the resized image\n",
    "startX = int(startX * w)\n",
    "startY = int(startY * h)\n",
    "endX = int(endX * w)\n",
    "endY = int(endY * h)\n",
    "\n",
    "# Draw the bounding box on the image\n",
    "cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)\n",
    "\n",
    "# Display the image with the bounding box\n",
    "plt.imshow(image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the path to the testing image\n",
    "imagePath = r\"\\\\192.168.239.77\\truenas\\image.jpg\"\n",
    "\n",
    "# Load and preprocess the testing image\n",
    "image = load_img(imagePath, target_size=(224, 224))\n",
    "image = img_to_array(image) / 255.0\n",
    "image = np.expand_dims(image, axis=0)\n",
    "\n",
    "# Make predictions using the trained model\n",
    "preds = model.predict(image)[0]\n",
    "\n",
    "# Extract predicted bounding box coordinates\n",
    "(startX, startY, endX, endY) = preds\n",
    "\n",
    "# Read the original image and resize it\n",
    "image = cv2.imread(imagePath)\n",
    "image = imutils.resize(image, width=600)\n",
    "(h, w) = image.shape[:2]\n",
    "\n",
    "# Scale the predicted coordinates to match the resized image\n",
    "startX = int(startX * w)\n",
    "startY = int(startY * h)\n",
    "endX = int(endX * w)\n",
    "endY = int(endY * h)\n",
    "\n",
    "# Draw the bounding box on the image\n",
    "cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)\n",
    "\n",
    "# Display the image with the bounding box\n",
    "plt.imshow(image)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "thermal",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
