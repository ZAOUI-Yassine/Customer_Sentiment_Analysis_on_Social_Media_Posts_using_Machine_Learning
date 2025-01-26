# Airline Tweets Sentiment Analysis

This project analyzes sentiment data from airline-related tweets to understand customer opinions and predict sentiments. Below is the summary of key steps, methods, and insights from the analysis.

---

## 1. Requirements
To run this notebook, install the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn nltk
```

---

## 2. Workflow Summary

### 2.1 **Preprocessing**
1. **Loading Data**:
   - Read the dataset `Tweets.csv`.
   - Initial exploration included data structure (`df.info()`), summary statistics, and visualizations.

2. **Data Cleaning**:
   - Removed duplicates.
   - Preprocessed text using custom functions to remove:
     - URLs, emojis, usernames, punctuation, and HTML tags.
     - Stopwords and contractions were handled.
     - Text was converted to lowercase and non-alphabetic characters were removed.

3. **Feature Engineering**:
   - Created a `tweet_length` column to analyze tweet lengths.
   - Combined `negativereason` with tweet text for enhanced context.
   - Vectorized text data using **TF-IDF**.

4. **Label Transformation**:
   - Converted sentiments (`positive`, `neutral`, `negative`) into numeric values (2, 1, 0).

5. **Handling Imbalance**:
   - Used **SMOTE** to balance class distributions.

---

### 2.2 **Exploratory Data Analysis (EDA)**
- Distribution of tweet lengths visualized using histograms.
- Sentiment analysis visualized via:
  - Bar plots and pie charts of sentiment distribution.
  - Reasons for negative tweets by airline.
- Distribution of tweets for each airline explored using bar plots.

---

### 2.3 **Modeling and Evaluation**
1. **Train-Test Split**:
   - Data split into 75% training and 25% testing sets.

2. **Models Evaluated**:
   - **SVM**: High accuracy and AUC.
   - **Random Forest**: Performed well with ensemble learning.
   - **Naive Bayes**: Limited due to the independence assumption.
   - **Decision Tree**: Prone to overfitting.
   - **KNN**: Lower accuracy compared to SVM and Random Forest.

3. **Evaluation Metrics**:
   - **Accuracy**: Computed for all models.
   - **Classification Report**: Detailed precision, recall, F1-score for each model.
   - **Confusion Matrices**: Visualized misclassifications.
   - **ROC and AUC**:
     - Plotted for models providing probabilities or decision functions.

---

### 2.4 **Key Insights**
- **Sentiment Analysis**:
  - Negative tweets dominate the dataset.
  - Neutral and positive sentiments are less frequent.
- **Airline Insights**:
  - Airlines like US Airways and United Airlines receive the highest negative sentiments.
  - Virgin America receives fewer negative sentiments comparatively.
- **Model Insights**:
  - SVM and Random Forest outperformed other models in accuracy and AUC scores.

---

## 3. Recommendations
- **Data Augmentation**:
  - Explore more synthetic techniques or expand dataset with newer tweets.
- **Improved Features**:
  - Use embeddings like Word2Vec, GloVe, or transformers (e.g., BERT) for semantic representation.
- **Hyperparameter Tuning**:
  - Perform grid search or randomized search for better model performance.
- **Advanced Models**:
  - Evaluate deep learning methods (LSTMs, GRUs, or transformers).
- **Explainability**:
  - Implement SHAP or LIME for model interpretability.

---

## 4. Results Summary
| Model              | Accuracy | Notable Observations            |
|--------------------|----------|---------------------------------|
| **SVM**           | High     | Strong AUC, good overall.       |
| **Random Forest** | High     | Robust ensemble model.          |
| **Naive Bayes**   | Medium   | Limited by independence assumption. |
| **Decision Tree** | Medium   | Overfitting observed.           |
| **KNN**           | Low      | Performance degraded with size. |

---

## 5. Visualizations
- Confusion Matrices for all models.
- ROC Curves for probabilistic models.
- Sentiment and airline-related distributions visualized with bar plots and pie charts.

---

Feel free to explore this repository further to add enhancements or use advanced methods!
