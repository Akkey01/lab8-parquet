"""
step6_csv_job.py
BD-1004 | Lab 8 | Spark job on transactions.csv — the baseline

Dataset : transactions.csv — 500,000 rows, 12 columns, 46 MB

Submit:
    spark-submit --deploy-mode client step6_csv_job.py

WRITE DOWN THE TIME. You will compare it in step8.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as spark_sum, max as spark_max
import os, time

spark = SparkSession.builder.appName("lab8_step6_csv").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

print("""
╔══════════════════════════════════════════════════════════════╗
║  Spark Job on transactions.csv — THE BASELINE                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  500,000 rows | 12 columns | 46 MB                           ║
║                                                              ║
║  What Spark must do with CSV:                                ║
║    - Read every byte of the 46 MB file from HDFS             ║
║    - Parse every character as text                           ║
║    - Cast every field to its type (string → float, etc.)     ║
║    - There is no way to skip columns or rows                 ║
║                                                              ║
║  WRITE DOWN THE TIME after this job finishes.                ║
║  You will run the same query on Parquet in step8.            ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Load ────────────────────────────────────────────────────────────────
print("Loading transactions.csv from HDFS...\n")
df = spark.read.csv(f"{BASE}/data/transactions.csv",
                    header=True, inferSchema=True)

df.printSchema()
print(f"Total rows : {df.count():,}")
print(f"Columns    : {len(df.columns)}\n")

# ── Show the data ────────────────────────────────────────────────────────
df.show(10)

# ── Benchmark query ─────────────────────────────────────────────────────
print("=" * 60)
print("RUNNING THE BENCHMARK QUERY")
print("=" * 60)
print("""
Query:
  For each region + category, compute:
    - number of transactions
    - total revenue
    - average amount
    - max amount

Only 3 of 12 columns are needed: region, category, amount.
CSV forces Spark to read and parse ALL 12 columns anyway.
""")

t_start = time.time()

result = df.select("region", "category", "amount") \
           .groupBy("region", "category") \
           .agg(
               count("*").alias("num_transactions"),
               spark_sum("amount").alias("total_revenue"),
               avg("amount").alias("avg_amount"),
               spark_max("amount").alias("max_amount")
           ) \
           .orderBy("region", "category")

result.show(30, truncate=False)

t_csv = time.time() - t_start

print("=" * 60)
print(f"  ⏱  CSV job time: {t_csv:.2f} seconds")
print("=" * 60)

print(f"""
Write this down: {t_csv:.2f}s

Next step: convert transactions.csv to Parquet (step7)
Then:      run the same query on Parquet (step8)
""")

# ── Show the plan — no column pruning, no pushdown possible ─────────────
print("Physical plan — notice there is no ReadSchema narrowing, no PushedFilters:")
df.select("region", "category", "amount") \
  .groupBy("region", "category").agg(count("*")) \
  .explain()

print("""
With CSV, Spark reads the full file every time.
No footer, no statistics, no way to skip anything.
""")

spark.stop()
