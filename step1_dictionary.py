"""
step1_dictionary.py
BD-1004 | Lab 8 | Encoding: Dictionary — Before & After

Dataset  : encoding_dictionary.csv
           5,000 rows | sale_id, region, category, payment_method, amount
           region       → 4 unique values
           category     → 4 unique values
           payment_method → 3 unique values
           amount       → thousands of unique floats

What you will see:
  BEFORE  — the raw CSV, cardinality per column
  CONVERT — write as Parquet
  AFTER   — file size on HDFS, parquet-tools meta showing encodings

Submit:
    spark-submit --deploy-mode client step1_dictionary.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct
import os

spark = SparkSession.builder.appName("lab8_step1_dictionary").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

# ── BEFORE ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Dictionary Encoding — BEFORE                                ║
╚══════════════════════════════════════════════════════════════╝

Dictionary encoding replaces repeated string values with small
integer IDs and stores the mapping (the dictionary) just once.

  Raw data :  North | South | North | North | East | South
  Dictionary: { North:0, South:1, East:2, West:3 }
  Encoded  :  0     | 1     | 0     | 0     | 2    | 1

  "Electronics" stored as a string = 11 bytes per row
  "Electronics" stored as integer  =  1 byte  per row
  Applied across 5,000 rows → 10x smaller on that column alone.

Best for: low-cardinality string columns (region, category, status)
NOT useful for: high-cardinality columns like amount (every value differs)
""")

df = spark.read.csv(f"{BASE}/data/encoding_dictionary.csv",
                    header=True, inferSchema=True)

print("Raw data (first 15 rows):")
df.show(15)
df.printSchema()
print(f"Total rows: {df.count():,}\n")

print("Unique values per column — this determines which encoding Parquet picks:")
df.agg(
    countDistinct("region").alias("unique_regions"),
    countDistinct("category").alias("unique_categories"),
    countDistinct("payment_method").alias("unique_payment_methods"),
    countDistinct("amount").alias("unique_amounts"),
).show()

print("Full value breakdown — region (4 values, 5,000 rows = heavy repetition):")
df.groupBy("region").agg(count("*").alias("count")) \
  .orderBy("count", ascending=False).show()

print("Full value breakdown — category:")
df.groupBy("category").agg(count("*").alias("count")) \
  .orderBy("count", ascending=False).show()

print(f"""CSV size on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_dictionary.csv
""")

# ── CONVERT ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Dictionary Encoding — CONVERTING to Parquet                 ║
╚══════════════════════════════════════════════════════════════╝
Parquet automatically picks Dictionary encoding for
region, category, payment_method (low cardinality strings).
It picks Plain for amount (high cardinality floats).
""")

df.write.mode("overwrite").option("compression", "snappy") \
  .parquet(f"{BASE}/parquet/encoding_dictionary.parquet")

print("Done. Parquet file written.\n")

# ── AFTER ──────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Dictionary Encoding — AFTER                                 ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Read the Parquet file back — same data, same schema:")
df2 = spark.read.parquet(f"{BASE}/parquet/encoding_dictionary.parquet")
df2.show(15)
df2.printSchema()
print(f"Total rows: {df2.count():,}\n")

print(f"""Compare sizes on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_dictionary.csv
    hdfs dfs -du -h {BASE}/parquet/encoding_dictionary.parquet/

Expected result:
    ~177 KB    encoding_dictionary.csv
    ~ 65 KB    encoding_dictionary.parquet   ← 0.37x (3x smaller)

The entire 3x reduction comes from Dictionary encoding.
No compression codec yet — just replacing repeated strings with integers.

Now inspect what Parquet actually wrote:
    hdfs dfs -get \\
      {BASE}/parquet/encoding_dictionary.parquet/part-00000-*.parquet \\
      dict_sample.parquet

    parquet-tools meta dict_sample.parquet

Look for:
    region         encodings: RLE_DICTIONARY   ← dictionary was used
    category       encodings: RLE_DICTIONARY   ← dictionary was used
    payment_method encodings: RLE_DICTIONARY   ← dictionary was used
    amount         encodings: PLAIN            ← no structure, raw bytes
""".format(BASE=BASE))

spark.stop()
