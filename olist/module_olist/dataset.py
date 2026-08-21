from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_data(
    orders_path: Path,
    items_path: Path,
    customers_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os dados de pedidos, itens e clientes a partir de arquivos CSV."""
    orders = pd.read_csv(orders_path, parse_dates=DATE_COLUMNS)
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)

    return orders, items, customers


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Salva um DataFrame em CSV, criando a pasta de destino se necessário."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.success(f"Dataset salvo na pasta: {output_path}")


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Seleciona pedidos entregues e cria o alvo binário de atraso."""
    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_approved_at"].notna()
    ].copy()

    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8")

    logger.info(f"Pedidos originais: {len(orders):,}")
    logger.info(f"Pedidos no recorte histórico: {len(delivered_orders):,}")
    logger.info(delivered_orders["is_late"].value_counts(dropna=False))

    return delivered_orders


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Agrega os itens para manter uma única linha por pedido."""
    items_agg = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )

    assert items_agg["order_id"].is_unique
    return items_agg


def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Cria a base analítica com uma linha por pedido."""
    orders = create_target(orders)
    items_agg = aggregate_items(items)

    data = orders.merge(
        items_agg,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    assert data["order_id"].is_unique
    assert len(data) == len(orders)

    return data


@app.command()
def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "dataset.csv",
) -> None:
    """Gera a base analítica usada pelo EDA e pelas etapas seguintes."""
    from module_olist.features import create_features

    orders, items, customers = load_data(orders_path, items_path, customers_path)
    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)
    save_dataset(dataset, output_path)


if __name__ == "__main__":
    app()
