### BERT-based YouTube Comment Analyzer for Influencers
This project is a high-performance NLP pipeline designed to analyze viewer sentiment on YouTube channels. It addresses the specific challenges of social media text, including noise and severe class imbalance, by leveraging fine-tuned transformer models and advanced resampling techniques.

Key Features
🧹 Robust Cleaning Pipeline: Processed a raw dataset of 40K+ comments, engineering custom features such as Comment Length and Stopword Density to improve baseline performance.

⚖️ Imbalance Handling (Classical): Tackled 3-class distribution skew for classical baselines by applying ADASYN (Adaptive Synthetic Sampling) on TF-IDF vectors.

🤖 Advanced Fine-Tuning: Fine-tuned bert-base-uncased with a Class-Weighted Loss function to prevent bias toward majority classes, optimizing a custom 3-node Softmax head.

📈 State-of-the-Art Accuracy: Achieved a Weighted F1-Score of 0.90 with the fine-tuned BERT model, outperforming classical baselines by 7%.

Tech Stack
Model: BERT (bert-base-uncased)

Techniques: Fine-Tuning, ADASYN, TF-IDF

Optimization: Class-Weighted Loss

Domain: NLP, Imbalanced Learning
