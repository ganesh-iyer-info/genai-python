import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Sample job description
job_description = """
Acme Corp is seeking a Senior Technical Program Manager to lead cloud modernization initiatives across enterprise platforms. The ideal candidate will have experience in Agile delivery, stakeholder alignment, and infrastructure transformation. Responsibilities include managing cross-functional teams, driving platform reliability, and coordinating with engineering and product leads. Familiarity with Python, AWS, and data center operations is preferred. Excellent communication and leadership skills are essential.
"""

# Preprocess
stop_words = set(stopwords.words('english'))

def clean(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

# Tokenize into sentences
sentences = sent_tokenize(job_description)
cleaned_sentences = [clean(s) for s in sentences]

# Vectorize
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

# Score sentences by TF-IDF sum
scores = tfidf_matrix.sum(axis=1).A1
ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

# Select top 3 sentences
summary = [s for s, score in ranked[:3]]

# Output
print("Job Description Summary:")
for line in summary:
    print("-", line)