# ===============================================
# Q12: Author Influence Network (Graph Analysis)
# ===============================================

# import sys
import os
import subprocess
import re


# 1. Clean conflicting variables
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

# 2. Auto-detect Java 17 (Mac specific)
try:
    cmd_output = subprocess.check_output(["/usr/libexec/java_home", "-v", "17"])
    java_home_path = cmd_output.strip().decode('utf-8')
    os.environ["JAVA_HOME"] = java_home_path
    print(f" JAVA_HOME set to: {java_home_path}")
except Exception:
    # Fallback
    manual_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    os.environ["JAVA_HOME"] = manual_path
    print(f" Using Fallback Java Path: {manual_path}")

# ==========================================

from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, udf, explode, desc
from pyspark.sql.functions import col, desc
from pyspark.sql.types import StringType, IntegerType, StructType, StructField

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("Author_Influence_Network") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print("\n Spark Graph Analysis Started")

# ==========================================
# 2. Load Data (Whole Text Files)
# ==========================================
# Hum 'wholeTextFiles' use kar rahe hain taaki header parse kar sakein
DATA_PATH = "../D184MB/*.txt"  
raw_rdd = sc.wholeTextFiles(DATA_PATH)

print(f"\n Files Detected: {raw_rdd.count()}")

# ==========================================
# 3. Preprocessing (Extract Author & Year)
# ==========================================

def extract_metadata(file_content):
    """
    Regex to find 'Author: ...' and 'Release Date: ...'
    """
    # 1. Extract Author
    author_match = re.search(r"Author:\s*(.+)", file_content)
    author = author_match.group(1).strip() if author_match else None
    
    # 2. Extract Year (4 digits inside Release Date line)
    # Pattern looks for "Release Date: [Month], [Year]"
    year_match = re.search(r"Release Date:.*?(\d{4})", file_content)
    year = int(year_match.group(1)) if year_match else None
    
    return (author, year)

# Apply Extraction
meta_rdd = raw_rdd.map(lambda x: extract_metadata(x[1])) \
                  .filter(lambda x: x[0] is not None and x[1] is not None)

# Convert to DataFrame
schema = StructType([
    StructField("Author", StringType(), True),
    StructField("Year", IntegerType(), True)
])

books_df = spark.createDataFrame(meta_rdd, schema).distinct()

print("\n Extracted Metadata:")
books_df.show(5, truncate=False)

# ==========================================
# 4. Construct Influence Network
# ==========================================
# Condition: Author A influenced Author B if:
# 1. A released book BEFORE or SAME YEAR as B
# 2. Within a window of X years (e.g., 10 years)

TIME_WINDOW = 10  # X years

# Self Join to create Edges
# Alias df1 = Influencer (Older), df2 = Influenced (Newer)
df1 = books_df.alias("df1")
df2 = books_df.alias("df2")

edges_df = df1.join(df2, col("df1.Author") != col("df2.Author")) \
    .where(
        (col("df2.Year") >= col("df1.Year")) & 
        ((col("df2.Year") - col("df1.Year")) <= TIME_WINDOW)
    ) \
    .select(
        col("df1.Author").alias("Influencer"),
        col("df1.Year").alias("Year_Start"),
        col("df2.Author").alias("Influenced"),
        col("df2.Year").alias("Year_End")
    )

print(f"\n Network Constructed (Window = {TIME_WINDOW} years)")
print("Edge Example (Influencer -> Influenced):")
edges_df.show(5)

# ==========================================
# 5. Analysis: In-Degree & Out-Degree
# ==========================================

# A. Out-Degree (Who influenced the most people?)
# Count how many times an author appears as 'Influencer'
out_degree_df = edges_df.groupBy("Influencer") \
    .count() \
    .withColumnRenamed("count", "Out_Degree") \
    .orderBy(desc("Out_Degree"))

# B. In-Degree (Who was influenced by the most people?)
# Count how many times an author appears as 'Influenced'
in_degree_df = edges_df.groupBy("Influenced") \
    .count() \
    .withColumnRenamed("count", "In_Degree") \
    .orderBy(desc("In_Degree"))

# ==========================================
# 6. Final Results
# ==========================================

print("\n" + "="*50)
print(" TOP 5 AUTHORS: HIGHEST OUT-DEGREE (Most Influential)")
print("="*50)
out_degree_df.show(5, truncate=False)

print("\n" + "="*50)
print(" TOP 5 AUTHORS: HIGHEST IN-DEGREE (Most Influenced)")
print("="*50)
in_degree_df.show(5, truncate=False)

# Clean Exit
spark.stop()
print(" Analysis Complete.")