# Lab 8: Parquet — Encoding, Compression & Spark

**BD-1004 | Big Data | NYU Center for Data Science**

---

## Setup

```bash
# 1. SSH into Dataproc
ssh $USER@dataproc.hpc.nyu.edu

# 2. Clone the repo
git clone https://github.com/<repo>/lab8-parquet.git
cd lab8-parquet

# 3. Upload all data to HDFS
hdfs dfs -mkdir -p /user/$USER/lab8/data
hdfs dfs -put data/ /user/$USER/lab8/

# 4. Confirm
hdfs dfs -ls /user/$USER/lab8/data/

# 5. Install parquet-tools (used to inspect file footers)
pip3 install parquet-tools --user
```

---

## Running the lab

```bash
# Part 1 — Encoding types (before & after for each)
spark-submit --deploy-mode client step1_dictionary.py
spark-submit --deploy-mode client step2_rle.py
spark-submit --deploy-mode client step3_delta.py
spark-submit --deploy-mode client step4_bitpack.py
spark-submit --deploy-mode client step5_plain.py

# Part 2 — The main act: CSV vs Parquet on 500k rows
spark-submit --deploy-mode client step6_csv_job.py       # ← write down the time
spark-submit --deploy-mode client step7_to_parquet.py    # ← convert + check sizes
spark-submit --deploy-mode client step8_parquet_job.py   # ← same job, compare
```

After every step:
```bash
hdfs dfs -du -h /user/$USER/lab8/
```
```
https://dataproc.hpc.nyu.edu/sparkhistory/
```

---

## Files

```
data/
  encoding_dictionary.csv    5,000 rows   region(4), category(4), payment(3)
  encoding_rle.csv           5,000 rows   is_fraud(95% False), status(skewed)
  encoding_delta.csv         5,000 rows   sequential IDs, evenly spaced timestamps
  encoding_bitpack.csv       5,000 rows   rating(1–5), items(1–10), weekday(1–7)
  encoding_plain.csv         5,000 rows   GPS coords, sensor readings (all unique floats)
  transactions.csv         500,000 rows   main dataset — all encoding types present

step1_dictionary.py     Before & after: low-cardinality strings
step2_rle.py            Before & after: skewed booleans and categoricals
step3_delta.py          Before & after: sequential IDs and timestamps
step4_bitpack.py        Before & after: small integer ranges
step5_plain.py          Before & after: high-cardinality floats (the contrast)
step6_csv_job.py        Spark job on transactions.csv — write down the time
step7_to_parquet.py     Convert to Parquet, all 4 codecs, hdfs du -h
step8_parquet_job.py    Same Spark job on Parquet — compare
```

---

## What to look for after each step

| Step | Run this after | What to see |
|---|---|---|
| step1 | `hdfs dfs -du -h .../encoding_dictionary*` | CSV ~177KB → Parquet ~65KB (0.37x) |
| step2 | `hdfs dfs -du -h .../encoding_rle*` | CSV ~151KB → Parquet ~63KB (0.42x) |
| step3 | `hdfs dfs -du -h .../encoding_delta*` | CSV ~197KB → Parquet ~52KB (0.26x) |
| step4 | `hdfs dfs -du -h .../encoding_bitpack*` | CSV ~78KB → Parquet ~40KB (0.51x) |
| step5 | `hdfs dfs -du -h .../encoding_plain*` | CSV ~254KB → Parquet ~249KB **(0.98x — barely anything)** |
| step6 | Write down the CSV time | Baseline for comparison |
| step7 | `hdfs dfs -du -h /user/$USER/lab8/` | All 4 codecs side by side |
| step8 | Compare to step6 time | `ReadSchema` shows 3 cols not 12 |

---

## Encoding quick reference

| Encoding | Best for | Example columns |
|---|---|---|
| Dictionary | Low-cardinality strings | region, category, status |
| RLE | Skewed booleans, repeated values | is_fraud (95% False), status |
| Delta | Sequential IDs, ordered timestamps | transaction_id, event_time |
| Bit-packing | Small integer ranges | rating (1–5), weekday (1–7) |
| Plain | High-cardinality floats | lat, lon, amount, sensor data |

---

## Most common mistake

```bash
# WRONG
spark-submit client step6_csv_job.py

# CORRECT
spark-submit --deploy-mode client step6_csv_job.py
```
