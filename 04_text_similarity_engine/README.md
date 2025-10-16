# 🔍 Project 4A: Text Similarity Engine

This project demonstrates how to compare a user query against a set of documents using TF-IDF vectorization and cosine similarity. It's a foundational NLP technique used in search engines, resume-job matching, and GenAI pipelines.

## 🧠 What It Does
- Cleans and tokenizes text using NLTK
- Converts text into numerical vectors using TF-IDF
- Computes similarity scores using cosine similarity
- Ranks documents based on relevance to the query

## 📥 Input
- A user query (e.g., "Looking for someone with Python and machine learning skills")
- A list of sample documents (e.g., resumes or job descriptions)

## 📤 Output
- Ranked list of documents based on similarity to the query

## 🔧 Tools Used
- Python 3.11
- NLTK (Natural Language Toolkit)
- Scikit-learn (TF-IDF and cosine similarity)

## 🚀 How to Run
```bash
pip install nltk scikit-learn
python text_similarity.py