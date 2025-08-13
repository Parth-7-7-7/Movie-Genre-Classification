# 🎬 Movie Genre Classification

## 📌 Project Overview
This project implements an **NLP-based text classification system** to predict **movie genres** from plot summaries. Using the **Genre Classification Dataset IMDb** from Kaggle, I applied **TF-IDF vectorization** and experimented with various supervised learning models. The best-performing model was **Logistic Regression**, achieving **58.6% accuracy**.

---

## 🗂 Dataset
- **Source:** [Kaggle - Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)  
- **Description:** Contains movie titles, plot summaries, and associated genres from IMDb.
- **Preprocessing:**
  - Lowercasing text
  - Removing punctuation & stopwords
  - Applying stemming/lemmatization
  - Using **n-grams** (unigrams, bigrams, trigrams)

---

## 🚀 Features
- Predicts **movie genres** based on plot summaries.
- Uses **TF-IDF vectorization** for text feature extraction.
- Experiments with multiple models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
- Performs **hyperparameter tuning** with `GridSearchCV` for optimization.
- Evaluates models with accuracy, precision, recall, and F1-score.

---

## 🛠 Technologies Used
- **Python 3**
- **Pandas, NumPy** – Data handling
- **Scikit-learn** – ML models, TF-IDF, GridSearchCV
- **Matplotlib, Seaborn** – Visualization
- **NLTK / spaCy** – Text preprocessing

---

## 📊 Model Workflow
1. **Data Loading & Preprocessing**
   - Text cleaning & tokenization
   - Stopword removal
   - Lemmatization
2. **Feature Engineering**
   - TF-IDF Vectorization
   - n-grams (1, 2, 3)
3. **Model Training & Tuning**
   - Logistic Regression
   - SVM
   - Naive Bayes
   - Hyperparameter tuning with GridSearchCV
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix

---

## 📈 Results
- **Best Model:** Logistic Regression
- **Accuracy:** 58.6%
- SVM and Naive Bayes had lower performance.
- n-grams improved performance slightly over unigrams.

---

## 📦 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/movie-genre-classification.git

# Navigate to project directory
cd movie-genre-classification

# Install dependencies
pip install -r requirements.txt

# Run the script
python genre_classification.py
