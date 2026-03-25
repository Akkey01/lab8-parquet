"""
step7_to_parquet.py
BD-1004 | Lab 8 | Convert transactions.csv → Parquet

Submit:
    spark-submit --deploy-mode client step7_to_parquet.py

After this job, run:
    hdfs dfs -du -h /user/$USER/lab8/
to see all sizes side by side.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
import os, time

spark = SparkSession.builder.appName("lab8_step7_convert").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

USER = os.environ["USER"]
BASE = f"hdfs:///user/{USER}/lab8"

print("""
╔══════════════════════════════════════════════════════════════╗
║  Converting transactions.csv → Parquet                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  We write four versions — one per compression codec:         ║
║    none   → encoding only, no codec                          ║
║    snappy → fast reads, moderate compression                 ║
║    gzip   → best ratio, slower reads                         ║
║    zstd   → near-gzip ratio, near-snappy speed               ║
║                                                              ║
║  This lets us see what ENCODING alone does (none)            ║
║  vs what ENCODING + CODEC together do (snappy/gzip/zstd)     ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Loading transactions.csv...")
df = spark.read.csv(f"{BASE}/data/transactions.csv",
                    header=True, inferSchema=True)
df = df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss"))

print(f"Loaded: {df.count():,} rows, {len(df.columns)} columns\n")
df.printSchema()

# ── Write all four codecs ───────────────────────────────────────────────
print("\nWriting Parquet files...\n")
codecs = ["none", "snappy", "gzip", "zstd"]
times  = {}

for codec in codecs:
    path = f"{BASE}/parquet/transactions_{codec}.parquet"
    t = time.time()
    df.write.mode("overwrite").option("compression", codec).parquet(path)
    times[codec] = time.time() - t
    print(f"  {codec:<8} written in {times[codec]:.1f}s")

print(f"""
All four files written.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOW RUN THIS on the master node:

    hdfs dfs -du -h {BASE}/

You will see something like:

    46.5 MB   data/transactions.csv          ← original
    21.8 MB   parquet/transactions_none      ← encoding only
    17.9 MB   parquet/transactions_snappy
    12.9 MB   parquet/transactions_gzip
    14.3 MB   parquet/transactions_zstd

What to notice:
  1. transactions_none is less than HALF the CSV
     → That is encoding doing the work before any codec runs
     → Dictionary on region/category/status/payment_type
     → Delta on transaction_id and event_time
     → RLE on is_flagged (95% zeros)
     → Bit-packing on quantity (1–10) and quarter (1–4)

  2. gzip gets the best ratio (12.9 MB = 28% of original)
     but is the slowest to write and read back

  3. zstd nearly matches gzip (14.3 MB) at much faster read speed
     → modern default for new pipelines

  4. snappy is the sweet spot for frequently-queried hot data
     → fast reads, reasonable compression, lowest CPU overhead
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Also check the metadata (footer) of the snappy file:

    hdfs dfs -get \\
      {BASE}/parquet/transactions_snappy.parquet/part-00000-*.parquet \\
      txn_sample.parquet

    parquet-tools meta txn_sample.parquet

What to look for per column:
    transaction_id  → DELTA_BINARY_PACKED         (sequential int)
    event_time      → DELTA_BINARY_PACKED         (ordered timestamp)
    region          → RLE_DICTIONARY              (5 unique values)
    category        → RLE_DICTIONARY              (6 unique values)
    status          → RLE_DICTIONARY              (3 unique values)
    payment_type    → RLE_DICTIONARY              (4 unique values)
    amount          → PLAIN                       (random floats)
    quantity        → RLE, BIT_PACKED             (range 1–10)
    is_flagged      → RLE, BIT_PACKED             (95% zeros)
    quarter         → RLE, BIT_PACKED             (range 1–4)
    lat / lon       → PLAIN                       (random floats)

Also look at compressed_size vs uncompressed_size per column:
    region   → tiny  (Dictionary crushed it)
    amount   → large (Plain — no encoding helped)
    is_flagged → tiny (RLE — 95% the same value)
""".format(BASE=BASE))

spark.stop()
