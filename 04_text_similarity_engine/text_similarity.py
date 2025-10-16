import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# Sample documents (e.g., resumes or job descriptions)
documents = [
    "Experienced Python developer with expertise in data analysis and machine learning.",
    "Cloud architect skilled in AWS, Azure, and infrastructure automation.",
    "Technical program manager with strong background in Agile and stakeholder alignment.",
    "Data scientist proficient in NLP, Scikit-learn, and sentiment analysis.",
    "DevOps engineer with experience in CI/CD, Docker, and Kubernetes."
]

# User query
query = "Looking for someone with Python and machine learning skills"

# Combine query and documents
all_texts = [query] + documents

# Preprocess: remove stopwords and punctuation
stop_words = set(stopwords.words('english'))

def clean(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

cleaned_texts = [clean(t) for t in all_texts]

# Vectorize
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

# Compute cosine similarity
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Rank results
ranked = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

# Display
print("Query:", query)
print("\nTop Matches:")
for doc, score in ranked:
    print(f"- ({score:.2f}) {doc}")