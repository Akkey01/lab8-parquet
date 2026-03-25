"""
step2_rle.py
BD-1004 | Lab 8 | Encoding: RLE — Before & After

Dataset  : encoding_rle.csv
           5,000 rows | txn_id, status, is_fraud, channel, amount
           status    → 3 values, heavily skewed (60% "completed")
           is_fraud  → boolean, 95% False
           channel   → 2 values (70% "online")

Submit:
    spark-submit --deploy-mode client step2_rle.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct
import os

spark = SparkSession.builder.appName("lab8_step2_rle").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

# ── BEFORE ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  RLE (Run-Length Encoding) — BEFORE                          ║
╚══════════════════════════════════════════════════════════════╝

RLE stores (value, count) pairs instead of repeating the same
value over and over.

  Raw    :  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ...
  RLE    :  (0, 9), (1, 1), (0, 6) ...

  is_fraud column: 95% zeros across 5,000 rows
  RLE encodes the whole column as roughly:
      (0, 4750), (1, 250)
  That is 2 entries instead of 5,000. Near-zero storage.

Best for: boolean columns, skewed categoricals, any column
          where the same value repeats in long runs
""")

df = spark.read.csv(f"{BASE}/data/encoding_rle.csv",
                    header=True, inferSchema=True)

print("Raw data (first 15 rows):")
df.show(15)
df.printSchema()
print(f"Total rows: {df.count():,}\n")

print("Value distribution — is_fraud (95% zeros = perfect for RLE):")
df.groupBy("is_fraud").agg(count("*").alias("count")) \
  .orderBy("is_fraud").show()

print("Value distribution — status (skewed toward 'completed'):")
df.groupBy("status").agg(count("*").alias("count")) \
  .orderBy("count", ascending=False).show()

print("Value distribution — channel (70% online):")
df.groupBy("channel").agg(count("*").alias("count")) \
  .orderBy("count", ascending=False).show()

print(f"""CSV size on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_rle.csv
""")

# ── CONVERT ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  RLE — CONVERTING to Parquet                                 ║
╚══════════════════════════════════════════════════════════════╝
Parquet applies RLE automatically to is_fraud, status, channel
because they have long runs of repeated values.
""")

df.write.mode("overwrite").option("compression", "snappy") \
  .parquet(f"{BASE}/parquet/encoding_rle.parquet")

print("Done. Parquet file written.\n")

# ── AFTER ──────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  RLE — AFTER                                                 ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Read the Parquet file back — same data, same schema:")
df2 = spark.read.parquet(f"{BASE}/parquet/encoding_rle.parquet")
df2.show(15)
print(f"Total rows: {df2.count():,}\n")

print(f"""Compare sizes on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_rle.csv
    hdfs dfs -du -h {BASE}/parquet/encoding_rle.parquet/

Expected result:
    ~151 KB    encoding_rle.csv
    ~ 63 KB    encoding_rle.parquet    ← 0.42x (2.4x smaller)

The is_fraud column goes from 5,000 entries → ~2 RLE pairs.
The amount column (random floats) stays large — no runs to exploit.
This is why column-level encoding matters: each column gets
the treatment that fits its own data shape.

Inspect the encodings:
    hdfs dfs -get \\
      {BASE}/parquet/encoding_rle.parquet/part-00000-*.parquet \\
      rle_sample.parquet

    parquet-tools meta rle_sample.parquet

Look for:
    is_fraud   → tiny compressed_size   (RLE worked — near-zero storage)
    status     → tiny compressed_size   (RLE + Dictionary)
    channel    → tiny compressed_size   (RLE + Dictionary)
    amount     → large compressed_size  (Plain — random floats, no runs)
""".format(BASE=BASE))

spark.stop()
