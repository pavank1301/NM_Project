# Spam Classifier

## Overview

This spam classifier is a machine learning model that can effectively distinguish between spam and non-spam messages. It uses natural language processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network to classify messages.

## Dependencies

Before running the code, make sure you have the following dependencies installed:

- Python (>=3.6)
- Jupyter Notebook
- scikit-learn
- TensorFlow
- Keras
- pandas
- numpy

You can install these dependencies using pip:

```bash
pip install scikit-learn tensorflow keras pandas numpy
```
## Usage

1.Clone the repository:
git clone [https://github.com/pavank1301/spam-classifier.git](https://github.com/pavank1301/spam_classifier.git)

2.Navigate to the project directory:
cd spam_classifier

3.Launch the Jupyter Notebook:
Open the Spam Classifier.ipynb notebook.

Run the notebook cell by cell to see the code and results.

To use the spam classifier model on your own dataset, modify the code accordingly in the notebook. Ensure your dataset follows the same format as the provided dataset.

### Dataset
The spam classifier uses a labeled dataset of spam and non-spam messages. The dataset is available at  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset. You can download and use this dataset for training and testing the model.

### Model
The spam classifier uses an LSTM-based model to analyze and classify messages. The model is trained on the provided dataset and achieves.

### Evaluation
You can find the model's evaluation metrics, including accuracy, precision, recall, F1-score, and the ROC-AUC curve, in the notebook.

### Future Improvements
Handle class imbalance issues in the dataset.
Implement real-time classification for email and messaging platforms.
Add multilingual support for messages in different languages.

## Spam Classifier

## Overview

This spam classifier is a machine learning model that effectively distinguishes between spam and non-spam messages. It uses natural language processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network to classify messages.

## Dataset

The spam classifier is trained on a publicly available dataset from the [SpamAssassin Project](https://spamassassin.apache.org/old/publiccorpus/). The dataset contains a collection of email messages labeled as either spam or non-spam (ham).

Please download the dataset from the following link:
 [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

After downloading, extract the dataset and organize it into two folders: "spam" and "ham" for spam and non-spam messages, respectively.

