"""
step5_plain.py
BD-1004 | Lab 8 | Encoding: Plain — Before & After

Dataset  : encoding_plain.csv
           5,000 rows | sensor_id, lat, lon, temperature, pressure, humidity
           All float columns have ~5,000 unique values — nearly every row differs.
           No encoding can help. Parquet falls back to Plain (raw bytes).

Submit:
    spark-submit --deploy-mode client step5_plain.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct
import os

spark = SparkSession.builder.appName("lab8_step5_plain").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

# ── BEFORE ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Plain Encoding — BEFORE                                     ║
╚══════════════════════════════════════════════════════════════╝

Plain encoding = no encoding. Raw bytes written as-is.
Parquet falls back to Plain when no encoding can help.

When does no encoding help?
  When every value is different — high precision floats like GPS
  coordinates, sensor readings, prices with many decimal places.

  lat = 40.712843, 34.051729, -12.043817, 51.509865, 19.432608 ...

  Dictionary? No — every value is unique, the dictionary would be
              as large as the data itself.
  RLE?        No — no repeated values, no runs.
  Delta?      No — differences between random floats are also random.
  Bit-pack?   No — these are full 64-bit floats.
  Plain:      Yes — just write the bytes.

The compression CODEC (snappy/gzip) still runs on top,
but without structural encoding first, gains are minimal.
""")

df = spark.read.csv(f"{BASE}/data/encoding_plain.csv",
                    header=True, inferSchema=True)

print("Raw data (first 15 rows):")
df.show(15)
df.printSchema()
print(f"Total rows: {df.count():,}\n")

total = df.count()
print("Unique value count per column (high = Plain encoding will be used):")
df.agg(
    countDistinct("lat").alias("unique_lat"),
    countDistinct("lon").alias("unique_lon"),
    countDistinct("temperature").alias("unique_temp"),
    countDistinct("pressure").alias("unique_pressure"),
    countDistinct("humidity").alias("unique_humidity"),
).show()

print(f"""Nearly every value is unique across {total:,} rows.
Dictionary, RLE, Delta, Bit-packing all have nothing to exploit.
Plain it is.

CSV size on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_plain.csv
""")

# ── CONVERT ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Plain Encoding — CONVERTING to Parquet                      ║
╚══════════════════════════════════════════════════════════════╝
""")

df.write.mode("overwrite").option("compression", "snappy") \
  .parquet(f"{BASE}/parquet/encoding_plain.parquet")

print("Done. Parquet file written.\n")

# ── AFTER ──────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  Plain Encoding — AFTER                                      ║
╚══════════════════════════════════════════════════════════════╝
""")

df2 = spark.read.parquet(f"{BASE}/parquet/encoding_plain.parquet")
df2.show(15)
print(f"Total rows: {df2.count():,}\n")

print(f"""Compare sizes on HDFS — run this now:
    hdfs dfs -du -h {BASE}/data/encoding_plain.csv
    hdfs dfs -du -h {BASE}/parquet/encoding_plain.parquet/

Expected result:
    ~254 KB    encoding_plain.csv
    ~249 KB    encoding_plain.parquet    ← 0.98x  (barely moved!)

Now compare that to what we saw in the previous steps:
    step1 Dictionary : CSV ~177 KB → Parquet  ~65 KB  (0.37x — 3x smaller)
    step2 RLE        : CSV ~151 KB → Parquet  ~63 KB  (0.42x — 2.4x smaller)
    step3 Delta      : CSV ~197 KB → Parquet  ~52 KB  (0.26x — 4x smaller)
    step4 Bit-pack   : CSV  ~78 KB → Parquet  ~40 KB  (0.51x — 2x smaller)
    step5 Plain      : CSV ~254 KB → Parquet ~249 KB  (0.98x — barely anything)

This is the key lesson:
  Parquet's power comes from encoding matching the data shape.
  When values have no structure, Parquet is just a typed binary
  format — still useful, but not dramatically smaller.

In real datasets like transactions.csv, MOST columns have structure.
That is why you see 4–5x overall compression on the full file.

Inspect the encodings:
    hdfs dfs -get \\
      {BASE}/parquet/encoding_plain.parquet/part-00000-*.parquet \\
      plain_sample.parquet

    parquet-tools meta plain_sample.parquet

Look for:
    lat, lon, temperature, pressure, humidity → PLAIN
    (and notice their compressed_size ≈ uncompressed_size — codec barely helped)
""".format(BASE=BASE))

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODING SUMMARY — all five types
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dictionary  → low-cardinality strings     (region, category, status)
  RLE         → skewed booleans/categoricals (is_fraud, channel)
  Delta       → sequential IDs, timestamps   (event_id, event_time)
  Bit-packing → small integer ranges         (rating 1–5, weekday 1–7)
  Plain       → high-cardinality floats      (lat, lon, sensor readings)

Parquet picks the best encoding per column automatically.
You never specify it — it just happens based on the data shape.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

spark.stop()
