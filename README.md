# 🧠 Emotion Detection from Tweets using Logistic Regression

This project uses machine learning to classify the emotional content of tweets.  
It is built using Logistic Regression with TF-IDF vectorization and SMOTE oversampling to handle class imbalance.

---

## 📁 Dataset

- **Source**: [Kaggle - Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
- **Columns**:
  - `tweet_id` → dropped
  - `sentiment` → renamed to `emotion`
  - `content` → renamed to `text`

---

## ⚙️ Preprocessing

- Lowercased all text
- Removed:
  - User handles (`@username`)
  - Stopwords
  - Special characters
- Filtered to **top 5 most frequent emotions**

---

## 🔍 Features

- `TF-IDF Vectorizer` with:
  - `max_features=5000`
  - `ngram_range=(1, 2)`
  - `stop_words="english"`

---

## 🧪 Model

- **Classifier**: `LogisticRegression(max_iter=1000)`
- **Oversampling**: `SMOTE` from `imbalanced-learn`
- **Evaluation**:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1)

---

## 📦 Installation

```bash
pip install neattext scikit-learn imbalanced-learn joblib
