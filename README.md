# 🎬 YouTube Comment Sentiment Analyzer (BERT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/Model-BERT--base--uncased-green)
![Status](https://img.shields.io/badge/Status-Maintained-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Decoding viewer sentiment at scale.** A high-performance NLP pipeline designed to help influencers understand their community by analyzing over 40K+ comments with state-of-the-art accuracy.

---

## 🚀 Overview

Influencers receive thousands of comments, but standard tools often fail to capture nuance or handle the noise of social media text. This project builds a **Fine-Tuned BERT model** specifically optimized for YouTube comments. 

Unlike generic sentiment analyzers, this pipeline tackles the **Class Imbalance Problem** (the tendency for comments to be overwhelmingly positive or neutral) using advanced resampling and loss-weighting techniques.

## ✨ Key Features

* **🧠 Deep Learning Core:** Fine-tuned `bert-base-uncased` with a custom 3-node Softmax head for precise sentiment classification.
* **⚖️ Imbalance Handling:**
    * **Classical approach:** Applied **ADASYN** (Adaptive Synthetic Sampling) on TF-IDF vectors to balance the dataset for baseline models.
    * **Deep Learning approach:** Implemented **Class-Weighted Loss** during BERT fine-tuning to prevent model bias toward majority classes.
* **🧹 Advanced Preprocessing:** Engineered a cleaning pipeline to handle emojis, slang, and noise, extracting custom features like **Comment Length** and **Stopword Density**.
* **📈 Proven Accuracy:** Achieved a **Weighted F1-Score of 0.90**, surpassing traditional machine learning baselines by **7%**.

---

## 🛠️ The Pipeline

1.  **Data Ingestion:** Processed **40,000+** raw YouTube comments.
2.  **Feature Engineering:** Extracted metadata (length, density) and cleaned text.
3.  **Baseline Modeling:** TF-IDF vectorization + ADASYN to handle the 3-class distribution skew.
4.  **Transformer Fine-Tuning:**
    * Base: `bert-base-uncased`
    * Optimization: Class-Weighted Loss Function
5.  **Inference:** Final classification into `Positive`, `Neutral`, or `Negative`.

---

## 🏆 Performance

| Model Type | Technique | Weighted F1-Score | Improvement |
| :--- | :--- | :--- | :--- |
| **Classical Baselines** | TF-IDF + ADASYN | ~0.84 | - |
| **Our Model (BERT)** | **Class-Weighted Loss** | **0.90** 🚀 | **+7%** |

*The fine-tuned BERT model effectively restricts generation to relevant sentiments and handles the "skew" of internet comments better than synthetic oversampling alone.*

---

## 💻 Tech Stack

* **Language:** Python
* **NLP:** HuggingFace Transformers (BERT), NLTK, Spacy
* **ML Libraries:** Scikit-learn (ADASYN, TF-IDF), PyTorch/TensorFlow
* **Data Handling:** Pandas, NumPy

## 📦 Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/YouTube-Sentiment-BERT.git](https://github.com/yourusername/YouTube-Sentiment-BERT.git)

# Install dependencies
pip install -r requirements.txt
