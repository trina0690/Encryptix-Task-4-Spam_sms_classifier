# 📩 Spam SMS Classification

Welcome to the **Spam SMS Classification** project! 🚀 This repository demonstrates various machine learning models designed to classify SMS messages as either **"ham"** (non-spam) or **"spam"**. Explore different algorithms to find the most effective method for this classification task.

Structure of the Dataset
Columns:

 v1: This column contains the labels for the messages. The possible values are:
*    ham: Indicates that the message is not spam.
*    spam: Indicates that the message is spam.

  v2: This column contains the actual text of the messages.
*    Unnamed: 2, Unnamed: 3, Unnamed: 4: These columns contain NaN values, suggesting that they are either placeholders for future data or unnecessary columns that can be dropped 

Dataset link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## 📊 Model Performance

Here’s a summary of the performance metrics for each model:

| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| **BernoulliNB**      | 98.1%    | 98.1%     | 98.1%   | 98.0%    |
| **MultinomialNB**    | 97.4%    | 97.5%     | 97.4%   | 97.3%    |
| **SVC**              | 97.1%    | 97.1%     | 97.1%   | 97.0%    |
| **LogisticRegression** | 95.5%  | 95.5%     | 95.5%   | 95.1%    |
| **GaussianNB**       | 87.2%    | 90.6%     | 87.2%   | 88.3%    |

### 🏅 Key Observations
- **BernoulliNB** shows the highest performance across all metrics, achieving an accuracy of 98.1% and an F1-Score of 98.0%. This indicates it is very effective at both precision and recall for spam detection.
- **MultinomialNB** and **SVC** also perform very well, with high accuracy and balanced precision and recall.
- **Logistic Regression** and **GaussianNB** follow, with slightly lower metrics, indicating they are less effective compared to the top models but still viable options.

## 🚀 Getting Started

### 🔧 Prerequisites
Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them using pip:
pip install numpy pandas scikit-learn matplotlib seaborn



🏃‍♂️ Usage
Clone the Repository:
git clone https://github.com/trina0690/Encryptix-Task-4-Spam_sms_classifier.git

Run the Model:

To train and evaluate models, execute:

python train_models.py

Model Files:

Trained models are saved as .pkl files. Example usage:

import joblib
from sklearn.naive_bayes import BernoulliNB

- Save the model
joblib.dump(BernoulliNB, 'bernoulli_nb_model.pkl')

- Load the model
loaded_model = joblib.load('bernoulli_nb_model.pkl')



## 🤝 Acknowledgements:

Kaggle for the dataset.

Encryptix for giving me this opportunity.

