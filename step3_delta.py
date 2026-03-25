"""
step3_delta.py
BD-1004 | Lab 8 | Encoding: Delta — Before & After

Dataset  : encoding_delta.csv
           5,000 rows | event_id, user_id, event_time, session_id, page_views
           event_id   → sequential integers 1, 2, 3, 4 ...
           user_id    → sequential integers 10001, 10002, 10003 ...
           event_time → evenly spaced timestamps, one per minute
           session_id → sequential integers 90001, 90002 ...

Submit:
    spark-submit --deploy-mode client step3_delta.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, unix_timestamp
from pyspark.sql.window import Window
import os

spark = SparkSession.builder.appName("lab8_step3_delta").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

# ── BEFORE ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Delta Encoding — BEFORE                                     ║
╚══════════════════════════════════════════════════════════════╝

Delta encoding stores the first value, then stores only the
DIFFERENCE between consecutive values instead of the values themselves.

  Sequential IDs:
    Raw  :  1,  2,  3,  4,  5,  6,  7 ...
    Delta:  1, +1, +1, +1, +1, +1, +1 ...
    → constant delta → stored as: first=1, delta=+1 (2 numbers total)
    → 5,000 integers stored as 2 numbers

  Timestamps (one event per minute):
    Raw  :  09:00, 09:01, 09:02, 09:03 ...
    Delta:  09:00,   +60,   +60,   +60 ...
    → constant delta → same trick → near-zero storage

Best for: sequential IDs, ordered timestamps, anything where
          consecutive differences are small or constant
""")

df = spark.read.csv(f"{BASE}/data/encoding_delta.csv",
                    header=True, inferSchema=True)

print("Raw data (first 15 rows):")
df.orderBy("event_id").show(15)
df.printSchema()
print(f"Total rows: {df.count():,}\n")

print("The sequential pattern — first and last few values:")
df.orderBy("event_id").select("event_id", "user_id", "session_id", "event_time") \
  .show(5)
df.orderBy(col("event_id").desc()) \
  .select("event_id", "user_id", "session_id", "event_time") \
  .show(5)

# Show actual deltas so students see what Parquet computes internally
print("Computing the actual deltas between consecutive rows:")
print("(This is what Parquet's Delta encoder does internally)\n")

w = Window.orderBy("event_id")
df.orderBy("event_id") \
  .withColumn("id_delta",
              col("event_id") - lag("event_id", 1).over(w)) \
  .withColumn("time_delta_sec",
              unix_timestamp("event_time") - lag(unix_timestamp("event_time"), 1).over(w)) \
  .select("event_id", "id_delta", "event_time", "time_delta_sec") \
  .filter(col("event_id").between(2, 10)) \
  .show()

print("""Every id_delta   = 1    → constant → stored once
Every time_delta = 60s  → constant → stored once

Instead of 5,000 × 8-byte integers for event_id,
Delta stores: first=1, delta=+1  →  2 numbers.
""")

print(f"""CSV size on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_delta.csv
""")

# ── CONVERT ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Delta Encoding — CONVERTING to Parquet                      ║
╚══════════════════════════════════════════════════════════════╝
Parquet automatically applies DELTA_BINARY_PACKED to
event_id, user_id, session_id (sequential integers)
and to event_time (stored internally as int64 microseconds).
""")

df.write.mode("overwrite").option("compression", "snappy") \
  .parquet(f"{BASE}/parquet/encoding_delta.parquet")

print("Done. Parquet file written.\n")

# ── AFTER ──────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Delta Encoding — AFTER                                      ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Read the Parquet file back — same data, same schema:")
df2 = spark.read.parquet(f"{BASE}/parquet/encoding_delta.parquet")
df2.orderBy("event_id").show(15)
print(f"Total rows: {df2.count():,}\n")

print(f"""Compare sizes on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_delta.csv
    hdfs dfs -du -h {BASE}/parquet/encoding_delta.parquet/

Expected result:
    ~197 KB    encoding_delta.csv
    ~ 52 KB    encoding_delta.parquet   ← 0.26x (almost 4x smaller)

The sequential ID and timestamp columns compress to almost nothing.
Delta encoding turns 5,000 integers into 2 numbers each.

Inspect the encodings:
    hdfs dfs -get \\
      {BASE}/parquet/encoding_delta.parquet/part-00000-*.parquet \\
      delta_sample.parquet

    parquet-tools meta delta_sample.parquet

Look for:
    event_id    → DELTA_BINARY_PACKED   ← Delta confirmed
    user_id     → DELTA_BINARY_PACKED
    session_id  → DELTA_BINARY_PACKED
    event_time  → DELTA_BINARY_PACKED   ← timestamps as int64 micros
    page_views  → RLE, BIT_PACKED       ← small range 1–20
""".format(BASE=BASE))

spark.stop()
