import joblib
from loguru import logger
import pandas as pd

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.active_learning import train_with_uncertainty_sampling
from module_olist.modeling.pipeline import create_gradient_boosting_pipeline
from module_olist.modeling.split import FEATURES, TARGET


def main():
    input_path = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
    data = pd.read_csv(input_path)
    X = data[FEATURES]
    y = data[TARGET]

    result = train_with_uncertainty_sampling(
        estimator=create_gradient_boosting_pipeline(),
        X=X,
        y=y,
        initial_size=min(500, len(X)),
        batch_size=250,
        max_labeled=min(2000, len(X)),
    )

    output_path = MODELS_DIR / "active_learning_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output_path)
    logger.success(
        f"Modelo active learning treinado com {len(result.labeled_indices)} rotulos "
        f"e salvo em: {output_path}"
    )


if __name__ == "__main__":
    main()