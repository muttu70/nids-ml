# Network Intrusion Detection System Using Machine Learning

## Project Overview

This project aims to build an **Intrusion Detection System (IDS)** for network traffic using **Machine Learning (ML) techniques**. The system classifies network traffic into either normal (benign) or anomalous (malicious) categories. By using machine learning models, the system can identify different types of network attacks like Denial of Service (DoS), Distributed Denial of Service (DDoS), Port Scanning, and other malicious activities.

The primary goal of the project is to develop an efficient, scalable, and accurate **Network Intrusion Detection System (NIDS)** that helps in detecting malicious network activity in real-time.

## Features

- **Real-time detection**: Identifies network intrusions as they happen.
- **Various ML Algorithms**: Uses supervised and unsupervised machine learning models.
- **Model Evaluation**: Multiple evaluation metrics to ensure the model's effectiveness.
- **Easy Deployment**: Ready-to-use solution with clear instructions.
- **Visualization**: Visualizes model performance and dataset characteristics.

## Table of Contents

1. [Technologies Used](#technologies-used)
2. [Data Collection](#data-collection)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## Technologies Used

- **Python 3.x**: The primary language for building the system.
- **Scikit-learn**: Used for implementing machine learning models such as Random Forest, SVM, Decision Trees, etc.
- **TensorFlow / Keras**: For training deep learning models (optional).
- **PyTorch**: Alternative deep learning framework (optional).
- **XGBoost**: Gradient boosting algorithm for optimized performance.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For handling numerical data and matrix operations.
- **Matplotlib** & **Seaborn**: For data visualization and evaluation plots.
- **Scapy**: For network traffic packet analysis.
- **Pyshark**: Python wrapper for Wireshark for analyzing captured network traffic.
- **Imbalanced-learn**: To handle class imbalance during training.
- **Joblib**: For saving and loading trained machine learning models.

## Data Collection

For the development of this intrusion detection system, several publicly available datasets are used to train and test the models. These datasets contain both normal and attack data samples. Some common datasets used in this project include:

- **KDD Cup 1999 Dataset**: The original dataset for intrusion detection tasks.
- **NSL-KDD Dataset**: A refined version of the KDD Cup 1999 dataset with improvements like removing duplicate records.
- **CICIDS (Canadian Institute for Cybersecurity)**: Modern dataset that contains both network and system logs with multiple attack scenarios.

The data consists of network traffic features such as:
- **Duration**: Duration of the connection.
- **Protocol Type**: Type of protocol (e.g., TCP, UDP).
- **Service**: The service used (e.g., HTTP, FTP).
- **Flag**: Type of connection (e.g., normal or attack).
- **Source IP and Destination IP**: Source and destination addresses.
- **Packet size, Time-to-live (TTL)**: Network-related features.

These features are then used to train the machine learning models to predict whether the traffic is normal or anomalous.

## System Architecture

The system follows a modular architecture, consisting of the following components:

1. **Data Collection and Preprocessing**:
   - Capture and collect network traffic data (using tools like **Scapy** and **Pyshark**).
   - Clean and preprocess data using **Pandas** and **NumPy** (e.g., handle missing values, normalize features).

2. **Feature Extraction**:
   - Extract relevant features from the network traffic data (e.g., packet size, duration, source/destination IP, protocol type, etc.).

3. **Machine Learning Models**:
   - Implement different machine learning algorithms, including:
     - **Supervised learning**: Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes.
     - **Deep Learning**: Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) (optional).
     - **Ensemble Methods**: XGBoost, LightGBM, etc.

4. **Evaluation and Tuning**:
   - Evaluate models using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
   - Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

5. **Real-Time Detection**:
   - Use the trained models to classify live network traffic and generate alerts for potential intrusions.

## Installation

### Prerequisites

- **Python 3.x**: Make sure Python 3 is installed on your system.
- **pip**: Python's package installer for managing dependencies.

### Steps to Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nids-ml.git
   cd nids-ml
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate  # For Windows
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Dependencies for Network Traffic Capture (Optional)

If you're planning to use real-time traffic capture, you may also need to install:
- **Wireshark**: For packet capture.
- **Scapy**: For network traffic manipulation.

## Usage

### 1. **Prepare the Dataset**

- Download the dataset (e.g., **KDD Cup 1999** or **CICIDS**).
- Use **Pandas** to load the dataset and handle any necessary preprocessing (e.g., dealing with missing values, normalizing features).

### 2. **Train the Model**

To train the machine learning model, run the following:
```bash
python train_model.py
```
This script will:
- Load the dataset.
- Preprocess the data.
- Train the selected machine learning model.

You can experiment with different algorithms by modifying the model selection in `train_model.py`.

### 3. **Detect Intrusions in Real-Time**

Once the model is trained, you can use it to classify live network traffic. To do so, run the following:
```bash
python detect_intrusions.py
```
This script will use the trained model to classify incoming traffic in real time, alerting you to any detected intrusions.

### 4. **Evaluate the Model**

To evaluate the performance of the trained model, run:
```bash
python evaluate_model.py
```
This will provide performance metrics such as:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## Evaluation

After training, the model’s performance is evaluated using metrics such as:
- **Accuracy**: The proportion of correct predictions made by the model.
- **Precision**: The proportion of true positives among the predicted positives.
- **Recall**: The proportion of true positives among actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve, which measures the trade-off between true positive rate and false positive rate.

## Contributing

Contributions are welcome! If you would like to improve or add features to this project, follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Implement your changes or fixes.
4. Submit a pull request.

Please ensure that you have tested your changes thoroughly.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/malmem-2022.html)
- [CICIDS Dataset](https://www.unb.ca/cic/datasets/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
    
