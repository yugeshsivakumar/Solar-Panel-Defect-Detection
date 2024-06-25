# Solar-Panel-Defect-Detection

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Creating Virtual Environments](#creating-virtual-environments)
    - [Physical Images Environment](#physical-images-environment)
    - [Thermal Images Environment](#thermal-images-environment)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
The Solar Panel Defect Detection project leverages machine learning to identify defects in solar panels using both physical and thermal images. This project aims to enhance the efficiency and maintenance of solar panels by providing an automated solution to detect defects early.

## Features
- **Dual Image Analysis**: Utilizes both physical and thermal images for comprehensive defect detection.
- **Machine Learning Models**: Implements various machine learning algorithms for accurate defect detection.
- **Automated Workflow**: Provides Jupyter Notebooks for data preprocessing, model training, and evaluation.
- **Visualization**: Includes visualization tools to inspect and understand the defects detected by the model.

## Installation

### Creating Virtual Environments

#### Physical Images Environment
Create a Conda virtual environment for physical images processing:

```bash
conda create --name physical_env python=3.9
conda activate physical_env
```

Install the dependencies for physical images processing:

```bash
git clone https://github.com/yugeshsivakumar/solar-panel-defect-detection.git
cd solar-panel-defect-detection/physical_images
conda install --file requirements.txt
```

#### Thermal Images Environment
Create a Conda virtual environment for thermal images processing:

```bash
conda create --name thermal_env python=3.9
conda activate thermal_env
```
Install the dependencies for thermal images processing:

```bash
cd ../thermal_images
conda install --file requirements.txt
```
## Usage

### Model Training
Train your machine learning model using the provided notebooks:

Open and execute train.ipynb to train the model using preprocessed data in the respective environments.
Defect Detection
Detect defects in new images using the trained model:

Open and execute detect.ipynb to perform defect detection on new physical and thermal images in their respective environments.

## Dataset
The dataset consists of physical and thermal images of solar panels. To obtain access to the dataset, please contact the project maintainer.

### How to Request the Dataset
To request access to the dataset, please send an email to:

Email: imyugesh.s@gmail.com

Include the following information in your email:

- Your full name
- Affiliation (e.g., university, company)
- Purpose of using the dataset

## Results
The results of the defect detection will be saved in the specified output directory, including visualizations and detailed reports.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

1. Fork the Project
    - Click on the 'Fork' button on the top right corner of this repository's page
   
2. Create your Feature Branch 
```bash
git checkout -b feature/AmazingFeature
```
3. Commit your Changes 
```bash
git commit -m 'Add some AmazingFeature'
```
4. Push to the Branch
```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request
   - Go to your forked repository on GitHub and click on 'New Pull Request'.
   - Fill out the Pull Request form with details about your proposed changes.

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Yugesh S - imyugesh.s@gmail.com

Project Link: https://github.com/yugeshsivakumar/solar-panel-defect-detection
