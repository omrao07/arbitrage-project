import re
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

class NLPModel:
    """
    NLP Model for extracting sentiment from news headlines, Fed speeches, and macro policy text.
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.classifier = LogisticRegression()
        self.trained = False

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def get_sentiment_score(self, text):
        """
        Returns VADER sentiment polarity score.
        Ranges from -1 (bearish) to +1 (bullish).
        """
        return self.vader.polarity_scores(text)['compound']

    def batch_sentiment_scores(self, texts):
        """
        Returns an array of sentiment scores for multiple headlines/sentences.
        """
        return np.array([self.get_sentiment_score(t) for t in texts])

    def train_classifier(self, texts, labels):
        """
        Train a custom classifier on labeled sentiment data (bullish/bearish/neutral).
        Labels: 1 = Bullish, 0 = Neutral, -1 = Bearish
        """
        cleaned = [self.clean_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned)
        y = np.array(labels)
        self.classifier.fit(X, y)
        self.trained = True

    def predict_label(self, text):
        """
        Predicts bullish/neutral/bearish from trained model.
        """
        if not self.trained:
            raise Exception("Model not trained yet.")
        cleaned = self.clean_text(text)
        X = self.vectorizer.transform([cleaned])
        return self.classifier.predict(X)[0]

    def classify_with_confidence(self, text):
        """
        Predicts and returns confidence score.
        """
        if not self.trained:
            raise Exception("Model not trained yet.")
        cleaned = self.clean_text(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.classifier.predict_proba(X)
        label = np.argmax(probs)
        confidence = np.max(probs)
        return label, confidence

# Example usage
if __name__ == "__main__":
    model = NLPModel()

    sample = [
        "Fed signals rate hike pause amid cooling inflation.",
        "US job market shows strength, boosting dollar outlook.",
        "IMF warns of recession risks in 2025."
    ]

    print("\n[Sentiment Scores]")
    for s in sample:
        print(f"'{s}' -> {model.get_sentiment_score(s):.3f}")