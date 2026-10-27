from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import shap

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import load_model

FEATURES = [
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

def main():
    refined_path = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
    input_path = refined_path if refined_path.exists() else INTERIM_DATA_DIR / "dataset.csv"
    data = pd.read_csv(input_path)
    X = data[FEATURES].sample(n=min(1000, len(data)), random_state=42)

    model, model_name, _ = load_model(
        model_path=MODELS_DIR / "best_model.joblib",
        metadata_path=MODELS_DIR / "metadata.json",
    )
    X_transformed = prepare_data_for_shap(model, X)
    explainer = create_explainer(model)
    shap_values = calculate_shap_values(explainer, X_transformed)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "shap_summary.png"
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(
        f"Explicabilidade gerada para {model_name} e salva em: {output_path}"
    )

if __name__ == "__main__":
    main()