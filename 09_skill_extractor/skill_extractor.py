import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample resume text
resume_text = """
Ganesh Iyer is a hands-on practitioner with expertise in Python, Agile, stakeholder alignment, cloud modernization, data center infrastructure, and platform reliability. He has developed with Scikit-learn, NLTK, Git, and CI/CD pipelines.
"""

# Define skill keywords
skill_keywords = [
    "Python", "Agile", "Git", "CI/CD", "Scikit-learn", "NLP", "cloud", "infrastructure",
    "stakeholder", "data center", "platform", "reliability", "NLTK", "program manager"
]

# Preprocess
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(resume_text.lower())
filtered_tokens = [t for t in tokens if t not in stop_words and t.isalpha()]

# Match skills
extracted_skills = set()
for skill in skill_keywords:
    pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
    if any(pattern.search(token) for token in filtered_tokens):
        extracted_skills.add(skill)

# Output
print("Extracted Skills:")
for skill in sorted(extracted_skills):
    print("-", skill)