import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.text_preprocessing import clean_text
df = pd. read_csv(r"C:\Users\hp\Desktop\sentiment analyses app\data\test (1).csv",encoding="cp1252")
df["text"] = df["text"].fillna("")
df["sentiment"] = df["sentiment"].fillna(df["sentiment"].mode()[0])
df["clean_text"] = df["text"].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
