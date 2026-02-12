from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum

spark = SparkSession.builder \
    .appName("PySparkExample") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ---------------- Sample Data ----------------
customers_data = [
    (1, "John", "USA"),
    (2, "Ram", "India"),
    (3, "Lisa", "USA")
]

orders_data = [
    (101, 1, 200),
    (102, 1, 300),
    (103, 2, 150),
    (104, 3, 400)
]

customers = spark.createDataFrame(customers_data, ["customer_id", "name", "country"])
orders = spark.createDataFrame(orders_data, ["order_id", "customer_id", "amount"])

# ---------------- DataFrame API ----------------
result_df = (customers
    .join(orders, "customer_id", "inner")
    .filter(col("amount") > 100)
    .groupBy("country")
    .agg(
        count("order_id").alias("total_orders"),
        sum("amount").alias("total_amount")
    )
    .filter(col("total_orders") > 1)
    .orderBy(col("total_amount").desc())
)

# ---------------- Spark SQL ----------------
customers.createOrReplaceTempView("customers")
orders.createOrReplaceTempView("orders")

result_sql = spark.sql("""
SELECT c.country,
       COUNT(o.order_id) AS total_orders,
       SUM(o.amount) AS total_amount
FROM customers c
JOIN orders o
  ON c.customer_id = o.customer_id
WHERE o.amount > 100
GROUP BY c.country
HAVING COUNT(o.order_id) > 1
ORDER BY total_amount DESC
""")

# Print ONLY ONE (choose one)
#result_sql.show()
result_df.show()   # <- uncomment this and comment result_sql.show() if you want DF output instead

spark.stop()
