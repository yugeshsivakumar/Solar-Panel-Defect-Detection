{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "gpu_available = tf.config.list_physical_devices('GPU')\n",
    "\n",
    "if gpu_available:\n",
    "    print(\"GPU is available\")\n",
    "  \n",
    "    config = tf.compat.v1.ConfigProto()\n",
    "    config.gpu_options.allow_growth = True\n",
    "    session = tf.compat.v1.Session(config=config)\n",
    "else:\n",
    "    print(\"GPU is not available\")\n",
    "\n",
    "# if gpu_available:\n",
    "#     session.close()\n",
    "#     print(\"GPU connection closed\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np \n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt \n",
    "from  pydotplus import graphviz\n",
    "from keras.utils.vis_utils import plot_model\n",
    "%matplotlib inline\n",
    "import random\n",
    "from cv2 import resize\n",
    "from glob import glob\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "from tensorflow.keras.preprocessing.image import load_img\n",
    "from tensorflow.keras.models import load_model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "img_height = 244\n",
    "img_width = 244\n",
    "train_ds = tf.keras.utils.image_dataset_from_directory(\n",
    "  'archive\\Faulty_solar_panel',\n",
    "  validation_split=0.2,\n",
    "  subset='training',\n",
    "  image_size=(img_height, img_width),\n",
    "  batch_size=32,\n",
    "  seed=42,\n",
    "  shuffle=True)\n",
    "print(\"----------------------------------------------------------------------\")\n",
    "val_ds = tf.keras.utils.image_dataset_from_directory(\n",
    "  'archive\\Faulty_solar_panel',\n",
    "  validation_split=0.2,\n",
    "  subset='validation',\n",
    "  image_size=(img_height, img_width),\n",
    "  batch_size=32,\n",
    "  seed=42,\n",
    "  shuffle=True)\n",
    "print(\"----------------------------------------------------------------------\")\n",
    "#model = load_model('solar_new.h5')\n",
    "new_model = tf.keras.models.load_model('solar_v1.h5')\n",
    "class_names = train_ds.class_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import numpy as np\n",
    "# import cv2\n",
    "# import tensorflow as tf\n",
    "# from tensorflow.keras.models import load_model\n",
    "# from tensorflow.keras.preprocessing import image\n",
    "\n",
    "# # Load the saved model\n",
    "# model = load_model('solar_v1.h5')\n",
    "\n",
    "# # Initialize the camera\n",
    "# cap = cv2.VideoCapture(0)  # 0 for the default camera\n",
    "\n",
    "# while True:\n",
    "#     # Capture frame-by-frame\n",
    "#     ret, frame = cap.read()\n",
    "    \n",
    "#     # Display the frame\n",
    "#     cv2.imshow('Camera', frame)\n",
    "    \n",
    "#     # Wait for user input (press 'q' to quit, 's' to save)\n",
    "#     key = cv2.waitKey(1)\n",
    "    \n",
    "#     if key == ord('q'):  # Quit\n",
    "#         break\n",
    "#     elif key == ord('s'):  # Save the image\n",
    "#         # Convert the frame to RGB (OpenCV uses BGR by default)\n",
    "#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "#         target_size = (244, 244)  # Provide the desired dimensions\n",
    "#         resized_image = tf.image.resize(rgb_frame, target_size)\n",
    "#         predictions = model.predict(tf.expand_dims(resized_image, 0))\n",
    "#         score = tf.nn.softmax(predictions[0])\n",
    "#         plt.imshow(rgb_frame)\n",
    "#         plt.title(\"Predicted:\"+ class_names[np.argmax(score)],fontdict={'color':'green'})\n",
    "        \n",
    "\n",
    "# # Release the camera and close the window\n",
    "# cap.release()\n",
    "# cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Via Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "image_path = r'\\\\192.168.239.77\\truenas\\imgpath.jpg'\n",
    "image = tf.io.read_file(image_path)\n",
    "image = tf.image.decode_image(image, channels=3)\n",
    "target_size = (244, 244) \n",
    "resized_image = tf.image.resize(image, target_size)\n",
    "predictions = new_model.predict(tf.expand_dims(resized_image, 0))\n",
    "score = tf.nn.softmax(predictions[0])\n",
    "plt.imshow(image)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "solar",
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
