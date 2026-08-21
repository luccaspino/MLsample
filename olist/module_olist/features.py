from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_olist.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

FEATURE_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria features temporais disponíveis no momento da previsão."""
    data = data.copy()

    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        (data["order_estimated_delivery_date"] - data["order_approved_at"])
        .dt.total_seconds()
        .div(86_400)
    )

    # Extrai o número do mês em que a compra foi realizada.
    data["purchase_month"] = data["order_purchase_timestamp"].dt.month

    # Pandas representa segunda-feira como 0 e domingo como 6.
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.dayofweek

    # Extrai a hora da compra, com valores de 0 a 23.
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour

    return data


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    """Gera as features temporais a partir da base analítica."""
    data = pd.read_csv(input_path, parse_dates=FEATURE_DATE_COLUMNS)
    features = create_features(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.success(f"Features salvas na pasta: {output_path}")


if __name__ == "__main__":
    app()
