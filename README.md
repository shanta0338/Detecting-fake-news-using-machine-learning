# 📰 Fake News Detection Using NLP and Machine Learning

This project focuses on **detecting fake news articles** using advanced Natural Language Processing (NLP) techniques and classical machine learning models. Leveraging **spaCy embeddings** and **ensemble learning**, the project achieves high accuracy in distinguishing between real and fake news.

---

## 📌 Project Overview

With the explosion of misinformation online, detecting fake news automatically has become crucial. This project presents a machine learning pipeline that converts news text into dense semantic vectors using **spaCy**, followed by classification using multiple supervised learning algorithms.

---

## 🗂️ Dataset

Source: [Kaggle - Fake News Dataset](https://www.kaggle.com/datasets/rajatkumar30/fake-news)

**Columns:**
- `Title`: Title of the news article.
- `Text`: Full content of the article.
- `Label`: Fake (1) or Real (0).

---

## 🧠 Technologies & Frameworks

- **Python 3**
- **Natural Language Processing (NLP)**
  - spaCy (Large Language Model for word embeddings)
- **Machine Learning**
  - Multinomial Naive Bayes (`MultinomialNB`)
  - K-Nearest Neighbors (`KNeighborsClassifier`)
  - Random Forest (`RandomForestClassifier`)
  - Gradient Boosting (`GradientBoostingClassifier`)
- **Preprocessing**
  - Text cleaning (lowercasing, punctuation removal, stopword removal)
  - Word vectorization (300-dimensional vectors via spaCy)
  - Feature stacking to form fixed-length input
- **Feature Scaling**
  - `MinMaxScaler` for normalization
- **Model Training**
  - `Pipeline`
  - `RandomizedSearchCV` for hyperparameter tuning
- **Evaluation**
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Classification Report

---

## ⚙️ Methodology

### 1. Data Preparation
- Clean text (lowercase, remove symbols, etc.)
- Binary label encoding (Fake = 1, Real = 0)

### 2. Feature Extraction
- Convert text into vectors using **spaCy large model**
- Stack vectors to 2D array suitable for ML input

### 3. Train-Test Split
- 80% training, 20% testing

### 4. Normalization
- Apply **MinMaxScaler** to ensure numerical stability

### 5. Model Building
#### Classifiers Used:
- **Multinomial Naive Bayes**
- **K-Nearest Neighbors**
- **Random Forest** (with hyperparameter tuning)
- **Gradient Boosting** (with hyperparameter tuning)

### 6. Hyperparameter Tuning
#### Random Forest Best Params:
```python
{
  "n_estimators": 200,
  "min_samples_split": 2,
  "min_samples_leaf": 2,
  "max_depth": 10
}
| Model               | Accuracy |
| ------------------- | -------- |
| Gradient Boosting   | 90.19%   |
| Random Forest       | 87.49%   |
| K-Nearest Neighbors | 82.00%   |
| Multinomial NB      | 76.00%   |

📌 Future Improvements
Incorporate deep learning models (LSTM, BERT)

Use multimodal inputs: combining text, metadata, and images

Deploy using Flask or Streamlit for real-time predictions
