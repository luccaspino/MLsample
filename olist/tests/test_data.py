import pandas as pd

from module_olist.dataset import create_dataset, create_target, aggregate_items
from module_olist.features import create_features


def test_create_target_builds_late_flag():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "order_status": ["delivered", "delivered", "shipped"],
            "order_delivered_customer_date": pd.to_datetime(
                ["2023-01-05", "2023-01-10", None]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2023-01-03", "2023-01-09", None]
            ),
            "order_approved_at": pd.to_datetime(["2023-01-01", "2023-01-08", None]),
            "order_purchase_timestamp": pd.to_datetime(
                ["2023-01-01", "2023-01-08", "2023-01-09"]
            ),
        }
    )

    result = create_target(orders)

    assert result["is_late"].tolist() == [1, 1]


def test_aggregate_items_groups_per_order():
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "seller_id": ["s1", "s2", "s1"],
            "price": [10.0, 5.0, 15.0],
            "freight_value": [1.0, 2.0, 3.0],
        }
    )

    result = aggregate_items(items)

    assert result["item_count"].tolist() == [2, 1]
    assert result["seller_count"].tolist() == [2, 1]
    assert result["total_price"].tolist() == [15.0, 15.0]


def test_create_features_adds_time_features():
    data = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_approved_at": pd.to_datetime(["2023-01-01"]),
            "order_estimated_delivery_date": pd.to_datetime(["2023-01-04"]),
            "order_purchase_timestamp": pd.to_datetime(["2023-01-02 14:30:00"]),
            "is_late": [1],
            "customer_id": ["c1"],
        }
    )

    result = create_features(data)

    assert "promised_days" in result.columns
    assert result["promised_days"].iloc[0] == 3.0
    assert result["purchase_month"].iloc[0] == 1
    assert result["purchase_weekday"].iloc[0] == 0
    assert result["purchase_hour"].iloc[0] == 14


def test_create_dataset_merges_all_sources():
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_status": ["delivered"],
            "order_delivered_customer_date": pd.to_datetime(["2023-01-05"]),
            "order_estimated_delivery_date": pd.to_datetime(["2023-01-03"]),
            "order_approved_at": pd.to_datetime(["2023-01-01"]),
            "order_purchase_timestamp": pd.to_datetime(["2023-01-01 09:00:00"]),
            "customer_id": ["c1"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_item_id": [1],
            "seller_id": ["s1"],
            "price": [25.0],
            "freight_value": [2.5],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "customer_city": ["São Paulo"],
            "customer_state": ["SP"],
        }
    )

    result = create_dataset(orders, items, customers)

    assert result["is_late"].iloc[0] == 1
    assert result["item_count"].iloc[0] == 1
    assert result["customer_state"].iloc[0] == "SP"
