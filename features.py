import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(df: pd.DataFrame):
    num_features = df[["tenure", "monthly_charges", "total_charges"]]

    cat_features = pd.get_dummies(
        df[["contract_type", "payment_method"]],
        drop_first=True
    )

    tfidf = TfidfVectorizer(
        max_features=300,
        stop_words="english"
    )
    text_features = tfidf.fit_transform(df["review_text"])

    X = pd.concat(
        [num_features, cat_features],
        axis=1
    )

    return X, text_features, df["churn"], tfidf
