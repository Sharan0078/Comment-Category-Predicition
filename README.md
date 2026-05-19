# Comment Category Classification using Classical Machine Learning

## Overview
This project focuses on multi-class text classification of online comments using classical machine learning techniques. The objective was to classify comments into different categories based on toxicity, aggression, political/religious discussions, and harmful behavior patterns.

The project was developed and evaluated using Kaggle, with the final solution achieving strong macro F1 performance using ensemble learning.

---

## Problem Statement
Online comment sections often contain harmful, toxic, or aggressive content. The goal of this project is to automatically classify comments into multiple categories using Natural Language Processing (NLP) and machine learning techniques.

---

## Dataset
The dataset contains text comments along with their corresponding class labels.

### Classes
- **Class 0** → Neutral / normal comments
- **Class 1** → Religion, race, gender, identity-related discussions
- **Class 2** → Political or controversial opinion-based comments
- **Class 3** → Aggressive, toxic, harmful, or highly offensive comments

---

## Project Workflow

### 1. Text Preprocessing
Basic preprocessing was applied to clean the text data:
- Lowercasing
- URL removal
- HTML tag removal
- Extra whitespace removal

---

### 2. Feature Engineering
Two different TF-IDF representations were combined:

#### Character-level TF-IDF (`char_wb`)
Used to capture:
- Aggressive writing styles
- Spelling variations
- Toxic word patterns
- Character-level signals

#### Word-level TF-IDF
Used to capture:
- Semantic meaning
- Contextual phrases
- Important word combinations

---

### 3. Feature Selection
Chi-Square feature selection was applied to retain the most informative features and reduce dimensionality.

---

## Models Used

### Logistic Regression
- Strong baseline model for sparse TF-IDF features
- Stable generalization performance

### MLP Classifier
- Captured nonlinear relationships in the feature space
- Achieved the best standalone validation performance

### LightGBM
- Used as an additional ensemble model
- Tuned carefully to reduce overfitting on sparse TF-IDF features

---

## Ensemble Learning
A soft voting ensemble was created using:
- Logistic Regression
- MLP Classifier
- LightGBM

The ensemble improved robustness and overall macro F1-score.

---

## Evaluation Metric
The project was evaluated using:

### Macro F1-Score
Macro F1-score was chosen because:
- The dataset is imbalanced
- All classes are equally important
- It balances precision and recall across classes

---

## Final Results

| Model | Validation Macro F1 |
|------|------|
| Logistic Regression | ~0.816 |
| MLP Classifier | ~0.822 |
| LightGBM | ~0.815 |
| Voting Ensemble | ~0.824 |

### Kaggle Test Score
**Final Kaggle Score: ~0.820**

---

## Technologies Used
- Python
- Scikit-learn
- LightGBM
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Key Learnings
- TF-IDF remains highly effective for classical NLP tasks
- Character n-grams are very useful for toxicity detection
- Ensemble learning improves generalization
- Tree-based models tend to overfit sparse TF-IDF features
- Proper regularization and feature engineering significantly impact macro F1 performance

---

## Future Improvements
Possible future improvements include:
- Transformer-based models (BERT, RoBERTa)
- Contextual embeddings
- Advanced ensemble techniques
- Hyperparameter optimization using Optuna
- Error analysis for class-2 and class-3 confusion

---

## Author
Sharan  
Machine Learning & NLP Project using Classical ML Techniques
