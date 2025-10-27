from data.utils import Data_utils
from config import SEED
import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

load_dotenv()

PIPELINE_PATH = os.getenv("PIPELINE_PATH")

def split_data_features(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y

def pipeline_preprocessing(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return pipeline

def split_train_test(X, y, test_size=0.2, random_state=SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)


if __name__ == "__main__":
    data = Data_utils()
    df = data.load_data()
    df = data.clean_data()
    X, y = split_data_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipeline = pipeline_preprocessing(X_train)
    save_pipeline(pipeline, X_train, y_train)
