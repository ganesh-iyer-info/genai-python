import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string

# Sample dataset
data = {
    'text': [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "Absolutely fantastic service.",
        "I hate it. Terrible quality.",
        "Very satisfied with the purchase.",
        "Not worth the money. Disappointed."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Preprocessing
def clean_text(text):
    # tokens = word_tokenize(text.lower())  will cause rare and misleading error—punkt_tab is not a real NLTK resource, and the issue is likely caused by a corrupted or misrouted installation of the punkt tokenizer.
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))