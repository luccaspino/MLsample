from pathlib import Path

from loguru import logger
import typer

from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features

app = typer.Typer()


@app.command()
def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "dataset.csv",
) -> None:
    """Carrega os dados brutos, cria as junções e salva a base intermediária."""
    logger.info("Carregando dados brutos...")
    orders, items, customers = load_data(orders_path, items_path, customers_path)

    logger.info("Criando a base analítica com as junções...")
    dataset = create_dataset(orders, items, customers)

    logger.info("Criando as features temporais...")
    dataset = create_features(dataset)
    save_dataset(dataset, output_path)

    logger.success(f"Pipeline concluído: {len(dataset):,} registros salvos em {output_path}")


if __name__ == "__main__":
    app()
