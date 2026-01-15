import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from features import build_features
from model import get_model
from evaluation import evaluate

def main():
    df = load_data("data/churn.csv")

    X_num_cat, X_text, y, tfidf = build_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num_cat)

    X_all = hstack([X_scaled, X_text])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_model()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print("Evaluation:", metrics)

if __name__ == "__main__":
    main()
