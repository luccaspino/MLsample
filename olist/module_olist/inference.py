from loguru import logger
import pandas as pd

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.predict import load_model, predict


def main():
    data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")
    features = [
        "promised_days",
        "item_count",
        "seller_count",
        "total_price",
        "total_freight",
        "purchase_month",
        "purchase_weekday",
        "purchase_hour",
        "customer_state",
    ]
    X = data[features]
    X_sample = X.sample(n=5, random_state=42)

    model, model_name, threshold = load_model(
        model_path=MODELS_DIR / "best_model.joblib",
        metadata_path=MODELS_DIR / "metadata.json",
    )
    predictions = predict(model=model, X=X_sample, threshold=threshold)

    logger.info(f"Amostras selecionadas para inferência:\n{X_sample}")
    logger.success(f"Predições realizadas pelo modelo {model_name}:\n{predictions}")


if __name__ == "__main__":
    main()