import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Sample FAQs
faq_data = {
    "How do I reset my password?": "Click 'Forgot Password' on the login page and follow the instructions.",
    "What is your refund policy?": "We offer a full refund within 30 days of purchase. Contact support for help.",
    "How can I contact customer service?": "You can reach us via email at support@example.com or call 1-800-123-4567.",
    "Do you offer international shipping?": "Yes, we ship to most countries. Shipping fees vary by location.",
    "Can I change my order after placing it?": "Yes, you can modify your order within 2 hours by contacting support."
}

# User query
user_question = "I want to change my order after buying"

# Prepare data
questions = list(faq_data.keys())
answers = list(faq_data.values())
all_texts = [user_question] + questions

# Vectorize
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Compute similarity
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
best_match_index = similarities.argmax()

# Output
print("User Question:", user_question)
print("\nBest Match:")
print("FAQ:", questions[best_match_index])
print("Answer:", answers[best_match_index])