# Olist - Dicionario Rapido dos CSVs

Este README resume os principais arquivos CSV do dataset Olist com nome, campos e utilidade.

## 1) olist_customers_dataset.csv

- Campos: `customer_id`, `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state`.
- Utilidade: identifica cada cliente e sua localizacao para ligar pedidos ao perfil geografico do comprador.

## 2) olist_geolocation_dataset.csv

- Campos: `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng`, `geolocation_city`, `geolocation_state`.
- Utilidade: fornece coordenadas e referencia geografica por CEP para analises espaciais e mapeamento de regioes.

## 3) olist_order_items_dataset.csv

- Campos: `order_id`, `order_item_id`, `product_id`, `seller_id`, `shipping_limit_date`, `price`, `freight_value`.
- Utilidade: detalha os itens de cada pedido, permitindo calcular receita, frete e composicao do carrinho por produto e vendedor.

## 4) olist_order_payments_dataset.csv

- Campos: `order_id`, `payment_sequential`, `payment_type`, `payment_installments`, `payment_value`.
- Utilidade: descreve como cada pedido foi pago, apoiando analises de meios de pagamento, parcelamento e valor transacionado.

## 5) olist_order_reviews_dataset.csv

- Campos: `review_id`, `order_id`, `review_score`, `review_comment_title`, `review_comment_message`, `review_creation_date`, `review_answer_timestamp`.
- Utilidade: registra avaliacao e comentarios dos clientes para medir satisfacao e investigar fatores que afetam a experiencia de compra.

## 6) olist_orders_dataset.csv

- Campos: `order_id`, `customer_id`, `order_status`, `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date`.
- Utilidade: e a tabela central do fluxo do pedido, usada para analisar status, prazos e performance logistica ponta a ponta.

## 7) olist_products_dataset.csv

- Campos: `product_id`, `product_category_name`, `product_name_lenght`, `product_description_lenght`, `product_photos_qty`, `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`.
- Utilidade: concentra atributos dos produtos para analise de catalogo, dimensoes fisicas e impacto nas vendas e no frete.

## 8) olist_sellers_dataset.csv

- Campos: `seller_id`, `seller_zip_code_prefix`, `seller_city`, `seller_state`.
- Utilidade: identifica vendedores e sua localizacao para estudos de cobertura geografica, marketplace e desempenho por regiao.

## 9) product_category_name_translation.csv

- Campos: `product_category_name`, `product_category_name_english`.
- Utilidade: traduz categorias de produto do portugues para ingles, facilitando analise e comunicacao em projetos internacionais.

## Relacionamentos Principais

- `orders.customer_id` conecta com `customers.customer_id`.
- `order_items.order_id`, `payments.order_id` e `reviews.order_id` conectam com `orders.order_id`.
- `order_items.product_id` conecta com `products.product_id`.
- `order_items.seller_id` conecta com `sellers.seller_id`.
- `customers.customer_zip_code_prefix` e `sellers.seller_zip_code_prefix` podem ser enriquecidos com `geolocation.geolocation_zip_code_prefix`.

