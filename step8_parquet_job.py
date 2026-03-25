"""
step8_parquet_job.py
BD-1004 | Lab 8 | Same Spark job on Parquet — compare to step6

Submit:
    spark-submit --deploy-mode client step8_parquet_job.py

This is the EXACT same query as step6.
Compare the time and the physical plan.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as spark_sum, max as spark_max
import os, time

spark = SparkSession.builder.appName("lab8_step8_parquet").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

print("""
╔══════════════════════════════════════════════════════════════╗
║  Same Spark Job — now on Parquet                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Identical query to step6. Only the source changes.          ║
║                                                              ║
║  What Spark can do now that it could NOT do with CSV:        ║
║                                                              ║
║  1. Column pruning                                           ║
║     Only region, category, amount are needed.                ║
║     Spark reads ONLY those 3 column chunks from HDFS.        ║
║     The other 9 (transaction_id, event_time, status, etc.)   ║
║     are never touched.                                       ║
║                                                              ║
║  2. Binary format                                            ║
║     No text parsing. amount is already float64.              ║
║                                                              ║
║  3. Compressed                                               ║
║     17.9 MB snappy vs 46.5 MB CSV — 2.6x less to read        ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Load Parquet ────────────────────────────────────────────────────────
print("Loading transactions_snappy.parquet from HDFS...\n")
df = spark.read.parquet(f"{BASE}/parquet/transactions_snappy.parquet")

df.printSchema()
print(f"Total rows : {df.count():,}")
print(f"Columns    : {len(df.columns)}\n")

df.show(10)

# ── Same benchmark query ────────────────────────────────────────────────
print("=" * 60)
print("RUNNING THE BENCHMARK QUERY  (same as step6)")
print("=" * 60)

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

t_parquet = time.time() - t_start

print("=" * 60)
print(f"  ⏱  Parquet job time: {t_parquet:.2f} seconds")
print("=" * 60)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPARISON

  Step 6 — CSV     : (your number from step6)
  Step 8 — Parquet : {t_parquet:.2f}s

Why Parquet is faster:
  - Reads 3 of 12 column chunks instead of the full file
  - Binary format → no text parsing
  - Compressed → 17.9 MB read instead of 46.5 MB
  - amount already typed as float64 → no casting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Physical plan — the proof ───────────────────────────────────────────
print("Physical plan — PARQUET:")
df.select("region", "category", "amount") \
  .groupBy("region", "category").agg(count("*")) \
  .explain()

print("""
Look for:
  ReadSchema: struct<region:string, category:string, amount:double>

Only 3 columns listed — NOT all 12.
The 9 other column chunks were never read from HDFS.

With CSV (step6), ReadSchema showed the full schema because
there is no way to skip columns in a text file.
""")

# ── Bonus: predicate pushdown ───────────────────────────────────────────
print("=" * 60)
print("BONUS — Predicate pushdown on Parquet")
print("=" * 60)
print("""
The Parquet footer stores min/max per column per row group.
Spark reads the footer first and skips row groups that
cannot possibly match your filter.
""")

t_start = time.time()
n = df.filter((col("amount") > 1800) & (col("region") == "North")).count()
t_filter = time.time() - t_start

print(f"Filter: amount > 1800 AND region = 'North'")
print(f"Matching rows : {n:,}")
print(f"Time          : {t_filter:.2f}s\n")

print("Physical plan for the filter — look for PushedFilters:")
df.filter((col("amount") > 1800) & (col("region") == "North")).explain()

print("""
PushedFilters: [IsNotNull(amount), IsNotNull(region),
                GreaterThan(amount,1800.0), EqualTo(region,North)]

Both filters are pushed into the Parquet reader.
Row groups where max(amount) < 1800 are skipped before
a single byte of data is decompressed.

With CSV this is impossible — every row is read and filtered
in Spark memory after the full file has been loaded.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check the Spark History Server:
    https://dataproc.hpc.nyu.edu/sparkhistory/

  SQL tab    → physical plan → ReadSchema (3 cols), PushedFilters
  Stages tab → fewer tasks on the Parquet job vs CSV job
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

spark.stop()
