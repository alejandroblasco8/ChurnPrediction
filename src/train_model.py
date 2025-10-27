from config import SEED
import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from data.utils import Data_utils
import src.data_preprocessing as dp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



data = Data_utils()
df = data.load_data()
df = data.clean_data()
X, y = dp.split_data_features(df)
X_train, X_test, y_train, y_test = dp.split_train_test(X, y, random_state=SEED)
pipeline = dp.pipeline_preprocessing(X_train)
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(random_state=SEED, max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=SEED, n_estimators=200)
}

results = {}

for name, model in models.items():
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

for name, metrics in results.items():
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


