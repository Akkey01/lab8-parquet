"""
step4_bitpack.py
BD-1004 | Lab 8 | Encoding: Bit-Packing — Before & After

Dataset  : encoding_bitpack.csv
           5,000 rows | order_id, rating, num_items, priority, is_returned, weekday
           rating      → 1–5   (needs 3 bits instead of 64)
           num_items   → 1–10  (needs 4 bits instead of 64)
           priority    → 1–3   (needs 2 bits instead of 64)
           is_returned → 0–1   (needs 1 bit  instead of 64)
           weekday     → 1–7   (needs 3 bits instead of 64)

Submit:
    spark-submit --deploy-mode client step4_bitpack.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, min as spark_min
import os, math

spark = SparkSession.builder.appName("lab8_step4_bitpack").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

# ── BEFORE ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Bit-Packing — BEFORE                                        ║
╚══════════════════════════════════════════════════════════════╝

A standard integer takes 64 bits in memory regardless of its value.
If values only range from 1 to 10, you only need 4 bits.
Parquet packs multiple small integers into a single 64-bit word.

  rating column: values 1–5
    Standard int64:  0000 0000 0000 0000 0000 0000 0000 0000
                     0000 0000 0000 0000 0000 0000 0000 0011  (= 3)
    3-bit packed:                                        011  (= 3)
    Saves 61 of 64 bits per value = 95% reduction

  is_returned column: 0 or 1
    1-bit packing → 64 values packed into one 64-bit word
    5,000 values → 79 words instead of 5,000 words

Best for: integer columns with a small known range
          (ratings, counts, boolean flags, day-of-week, priority levels)
""")

df = spark.read.csv(f"{BASE}/data/encoding_bitpack.csv",
                    header=True, inferSchema=True)

print("Raw data (first 15 rows):")
df.show(15)
df.printSchema()
print(f"Total rows: {df.count():,}\n")

int_cols = ["rating", "num_items", "priority", "is_returned", "weekday"]
stats = df.select(
    *[spark_min(c).alias(f"min_{c}") for c in int_cols],
    *[spark_max(c).alias(f"max_{c}") for c in int_cols],
).collect()[0]

print("Value ranges and bits needed per column:")
print(f"  {'Column':<15} {'Min':>5} {'Max':>5} {'Bits needed':>12} {'Savings vs int64':>18}")
print(f"  {'-'*58}")
for c in int_cols:
    mn   = stats[f"min_{c}"]
    mx   = stats[f"max_{c}"]
    bits = math.ceil(math.log2(mx + 1)) if mx > 0 else 1
    saving = round((1 - bits / 64) * 100, 1)
    print(f"  {c:<15} {mn:>5} {mx:>5} {bits:>12} bits {saving:>16}%")

print()
print("Distribution — rating:")
df.groupBy("rating").count().orderBy("rating").show()

print("Distribution — is_returned:")
df.groupBy("is_returned").count().orderBy("is_returned").show()

print(f"""CSV size on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_bitpack.csv
""")

# ── CONVERT ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Bit-Packing — CONVERTING to Parquet                         ║
╚══════════════════════════════════════════════════════════════╝
Parquet automatically applies BIT_PACKED to all integer columns
with small ranges. Each column uses only the bits it actually needs.
""")

df.write.mode("overwrite").option("compression", "snappy") \
  .parquet(f"{BASE}/parquet/encoding_bitpack.parquet")

print("Done. Parquet file written.\n")

# ── AFTER ──────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Bit-Packing — AFTER                                         ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Read the Parquet file back — same data, decoded automatically:")
df2 = spark.read.parquet(f"{BASE}/parquet/encoding_bitpack.parquet")
df2.show(15)
print(f"Total rows: {df2.count():,}\n")

print(f"""Compare sizes on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_bitpack.csv
    hdfs dfs -du -h {BASE}/parquet/encoding_bitpack.parquet/

Expected result:
    ~78 KB    encoding_bitpack.csv
    ~40 KB    encoding_bitpack.parquet   ← 0.51x (2x smaller)

Every integer column is packed to its minimum bit width.
is_returned (1 bit) and priority (2 bits) are especially compact.

Inspect the encodings:
    hdfs dfs -get \\
      {BASE}/parquet/encoding_bitpack.parquet/part-00000-*.parquet \\
      bitpack_sample.parquet

    parquet-tools meta bitpack_sample.parquet

Look for:
    rating      → RLE, BIT_PACKED   ← bit-packing confirmed
    num_items   → RLE, BIT_PACKED
    priority    → RLE, BIT_PACKED
    is_returned → RLE, BIT_PACKED   ← 1-bit packing
    weekday     → RLE, BIT_PACKED
""".format(BASE=BASE))

spark.stop()
