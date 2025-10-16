import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample resume
resume_text = """
Transformation leader and technical program manager with 15+ years of experience in cloud modernization, data center infrastructure, and enterprise delivery across financial, telecom, healthcare, and tech sectors. Skilled in Python, Agile, stakeholder alignment, and platform reliability.
"""

# Sample job descriptions
job_descriptions = [
    "Seeking a cloud architect with experience in AWS, Azure, and infrastructure automation.",
    "Looking for a technical program manager with strong Agile delivery and stakeholder engagement skills.",
    "Hiring a data scientist with expertise in NLP, Scikit-learn, and sentiment analysis.",
    "We need a DevOps engineer with CI/CD, Docker, and Kubernetes experience.",
    "Searching for a transformation leader to drive enterprise modernization and platform resilience."
]

# Combine resume and jobs
all_texts = [resume_text] + job_descriptions

# Preprocess
stop_words = set(stopwords.words('english'))

def clean(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

cleaned_texts = [clean(t) for t in all_texts]

# Vectorize
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

# Compute similarity
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Rank results
ranked = sorted(zip(job_descriptions, similarities), key=lambda x: x[1], reverse=True)

# Display
print("Resume Summary:")
print(resume_text)
print("\nTop Matching Jobs:")
for job, score in ranked:
    print(f"- ({score:.2f}) {job}")