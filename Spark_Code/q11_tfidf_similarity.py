# ===============================
# TF-IDF Book Similarity (FINAL FIXED VERSION)
# Akshat Jain
# ===============================

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name,
    lower,
    regexp_replace,
    collect_list,
    concat_ws,
    size
)

from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    Normalizer
)


# ===============================
# 1. Java Fix
# ===============================

os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home"


# ===============================
# 2. Start Spark
# ===============================

spark = SparkSession.builder \
    .appName("TFIDF_Book_Similarity") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n Spark Started Successfully")


# ===============================
# 3. Load All Books
# ===============================

DATA_PATH = "../D184MB/*.txt"

raw = spark.read.text(DATA_PATH) \
    .withColumn("file_name", input_file_name())

print("\n Raw Data Loaded")

raw.select("file_name").distinct().show(10, truncate=False)


# ===============================
# 4. Merge Lines Per File (FIXED)
# ===============================

docs = raw.groupBy("file_name") \
    .agg(
        concat_ws(" ", collect_list("value")).alias("text")
    )

# Take only 10 books
docs = docs.limit(50)

print("\n Documents Merged")

docs.select("file_name").show(truncate=False)


# ===============================
# 5. Cleaning
# ===============================

clean_df = docs.withColumn(
    "clean_text",
    lower(regexp_replace("text", "[^a-zA-Z ]", " "))
)

print("\n Preprocessing Output")

clean_df.select("file_name", "clean_text").show(2, truncate=100)


# ===============================
# 6. Tokenize
# ===============================

tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="tokens"
)

token_df = tokenizer.transform(clean_df)


# ===============================
# 7. Remove Stopwords
# ===============================

remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)

filtered_df = remover.transform(token_df)

print("\n After Stopword Removal")

filtered_df.select(
    "file_name",
    size("filtered_tokens").alias("word_count")
).show()


# ===============================
# 8. TF
# ===============================

hash_tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="tf_features",
    numFeatures=1000
)

tf_df = hash_tf.transform(filtered_df)

print("\n TF Sample")

tf_df.select("file_name", "tf_features").show(2, truncate=50)


# ===============================
# 9. TF-IDF
# ===============================

idf = IDF(
    inputCol="tf_features",
    outputCol="tfidf_features"
)

idf_model = idf.fit(tf_df)

tfidf_df = idf_model.transform(tf_df)

print("\n TF-IDF Sample")

tfidf_df.select("file_name", "tfidf_features").show(2, truncate=50)


# ===============================
# 10. Normalize
# ===============================

normalizer = Normalizer(
    inputCol="tfidf_features",
    outputCol="norm_features"
)

norm_df = normalizer.transform(tfidf_df)

print("\n Normalized Sample")

norm_df.select("file_name", "norm_features").show(2, truncate=50)


# ===============================
# 11. Collect Data
# ===============================

data = norm_df.select(
    "file_name",
    "norm_features"
).collect()

print("\n Vectors Collected")


# ===============================
# 12. Cosine Similarity
# ===============================

def cosine_sim(v1, v2):

    # Convert to dict for fast lookup
    d1 = dict(zip(v1.indices, v1.values))
    d2 = dict(zip(v2.indices, v2.values))

    # Dot product
    dot = 0.0

    for i in d1:
        if i in d2:
            dot += d1[i] * d2[i]

    # Norms
    norm1 = sum(x*x for x in d1.values()) ** 0.5
    norm2 = sum(x*x for x in d2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)



# ===============================
# 13. Target Book
# ===============================

target_file, target_vec = data[0]

print("\n Target Book:")
print(target_file.split("/")[-1])


# ===============================
# 14. Compute Similarity
# ===============================

results = []

for name, vec in data:

    if name != target_file:

        sim = cosine_sim(vec, target_vec)

        results.append((name, sim))


# ===============================
# 15. Top-5
# ===============================

top5 = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[:5]


# ===============================
# 16. Output
# ===============================

print("\n" + "="*70)
print(" TOP 5 MOST SIMILAR BOOKS")
print("="*70)

for i, (name, score) in enumerate(top5, 1):

    short = name.split("/")[-1]

    print(f"\n{i}. File : {short}")
    print(f"   Score: {round(score,4)}")

print("\n" + "="*70)
print(" Finished Successfully")
print("="*70)


# ===============================
# 17. Stop Spark
# ===============================

spark.stop()
