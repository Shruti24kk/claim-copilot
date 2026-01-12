from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from .synthetic_data import generate

texts, y = generate()

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

art = Path(__file__).parent / "artifacts"
art.mkdir(parents=True, exist_ok=True)

joblib.dump(tfidf, art / "tfidf.joblib")
joblib.dump(model, art / "text_model.joblib")

print("Model trained and saved.")
