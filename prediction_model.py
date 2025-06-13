import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib


def load_data(path: str) -> pd.DataFrame:
    """Load the competition data from CSV."""
    return pd.read_csv(path)


def build_preprocessor(categorical_cols, numeric_cols):
    """Create preprocessing pipeline for categorical and numeric features."""
    transformers = []
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    return ColumnTransformer(transformers)


def train_model(df: pd.DataFrame):
    """Train a RandomForestRegressor on the given DataFrame."""
    features = [
        "Store_ID",
        "Item_ID",
        "Price",
        "Item_Quantity",
        "Competition_Price",
    ]
    target = "Sales_Amount"
    X = df[features]
    y = df[target]

    categorical_cols = ["Store_ID", "Item_ID"]
    numeric_cols = ["Price", "Item_Quantity", "Competition_Price"]

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error: {mae:.2f}")

    return model, preprocessor


def save_model(model, preprocessor, path: str):
    """Save model and preprocessor to disk."""
    joblib.dump({"model": model, "preprocessor": preprocessor}, path)


def main():
    df = load_data("data/raw/Competition_Data.csv")
    model, preprocessor = train_model(df)
    save_model(model, preprocessor, "model.pkl")


if __name__ == "__main__":
    main()
