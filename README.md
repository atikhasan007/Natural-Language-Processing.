# Natural Language Processing (NLP) Overview

## 1. Algorithms Used in NLP

### a. Machine Learning Algorithms
- Naive Bayes
- Support Vector Machines (SVM)
- Logistic Regression
- Decision Trees
- Random Forest

### b. Deep Learning Algorithms
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Convolutional Neural Networks (CNN)
- Transformers

### c. Large Language Models (LLMs)
- GPT-3
- BERT
- T5
- RoBERTa
- XLNet

---

## 2. Tools Used in NLP

- **Python Libraries:**
  - NLTK
  - TextBlob
  - re (Regular Expressions)
- **Machine Learning Libraries:**
  - scikit-learn
  - Huggingface Transformers
  - Gensim
  - Keras (TensorFlow)

---

## 3. Applications of NLP

- Recommendation Systems
- Chatbots
- Social Media Analysis
- Search Engines
- Speech Recognition
- Machine Translation
- Spam Detection
- Text Generation
- Topic Modeling
- Summarization

---

## 4. Dataset Types

- Text-based datasets
- Audio-based datasets

---

## 5. NLP Project Pipeline

1. **Data Collection**
   - Sources:
     - Kaggle
     - UCI Machine Learning Repository
     - Company Databases
     - APIs

2. **Text Cleaning (Preprocessing)**
   - Lowercasing
   - Removing Stopwords
   - Punctuation Removal
   - Removing Numbers
   - Tokenization
   - Stemming
   - Lemmatization
   - Removing Whitespace
   - Removing Special Characters
   - Removing HTML Tags
   - Removing URLs

3. **Train-Test Split**

4. **Feature Extraction (Text to Numeric)**
   - Bag of Words (BoW)
   - TF-IDF Vectorizer
   - Word2Vec
   - GloVe
   - FastText
   - One-hot Encoding
   - Count Vectorizer
   - Hashing Vectorizer
   - BERT Embeddings
   - Doc2Vec
   - Sentence-BERT
   - Latent Dirichlet Allocation (LDA)
   - Latent Semantic Analysis (LSA)

5. **Model Training**

6. **Model Evaluation**

7. **Model Deployment**

===========================================================================================
# Sarcasm Detection Data Cleaning

This project loads a sarcasm tweet dataset and preprocesses the tweet texts to prepare them for model training.

## Features

- Loads dataset containing tweets and corresponding sarcasm labels.
- Handles and removes missing values.
- Cleans text by removing:
  - Emails
  - URLs
  - Emojis and non-ASCII characters
  - Punctuation
  - Stopwords
- Expands contractions (e.g., "can't" → "cannot") and normalizes slang.
- Applies lemmatization to convert words to their root forms.

## Usage

1. Place the `Sarcasm.csv` file in your working directory.
2. Run the notebook or script to clean and preprocess the data.
3. Use the cleaned tweets for sarcasm classification or other NLP tasks.

---

Feel free to customize or extend this README as your project grows!
