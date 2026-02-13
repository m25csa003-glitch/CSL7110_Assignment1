# CSL7110: Big Data Systems - Assignment 1
### Hadoop MapReduce & Apache Spark Analysis

**Student Details:**
- **Name:** Akshat Jain
- **Roll No:** M25CSA003
- **Instructor:** Dr. Dip Sankar Banerjee
- **Institute:** IIT Jodhpur

---

## Project Overview
This repository contains the implementation of distributed data processing tasks using **Hadoop MapReduce (Java)** and **Apache Spark (PySpark)**. The assignment covers text analytics, metadata extraction, and graph-based network analysis using the Project Gutenberg dataset.

---

## Tech Stack
- **Frameworks:** Apache Hadoop 3.4.2, Apache Spark 3.x
- **Languages:** Java (MapReduce), Python (PySpark)
- **Environment:** Single Node Cluster (macOS ARM64)

---

## Detailed Task List & Solutions

### Part 1: Apache Hadoop & MapReduce (Q1 - Q9)

| Task | Description |
| :--- | :--- |
| **Q1** | **WordCount Execution:** Successfully set up a single-node cluster and executed the standard WordCount example. |
| **Q2** | **Map Phase Analysis:** Traced (Key, Value) pairs for song lyrics. Identified Hadoop-specific data types (`LongWritable`, `Text`). |
| **Q3** | **Reduce Phase Analysis:** Analyzed how shuffled `(Key, Iterable<Value>)` pairs are processed by the Reducer. |
| **Q4** | **Data Type Mapping:** Replaced placeholders in `WordCount.java` with correct Hadoop IO classes. |
| **Q5** | **Custom Mapper:** Implemented a Mapper using `StringTokenizer` and `replaceAll()` to handle punctuation and case-insensitivity. |
| **Q6** | **Custom Reducer:** Developed logic to aggregate word frequencies and handle the final context writing. |
| **Q7** | **HDFS Operations:** Performed operations on `200.txt`. Analyzed replication factors and directory management in HDFS. |
| **Q8** | **HDFS Listing:** Explored `ls` output, understanding block replication and storage metadata. |
| **Q9** | **Performance Tuning:** Modified code to measure execution time and experimented with `split.maxsize` for performance optimization. |

### Part 2: Apache Spark Data Analysis (Q10 - Q12)

| Task | Description |
| :--- | :--- |
| **Q10** | **Metadata Extraction:** Used Regex to extract `Title`, `Release Date`, `Language`, and `Encoding` from raw book text. Performed year-wise analysis. |
| **Q11** | **TF-IDF & Similarity:** Cleaned text data, calculated Term Frequency (TF) and Inverse Document Frequency (IDF), and identified top 5 similar books using **Cosine Similarity**. |
| **Q12** | **Author Influence Network:** Constructed a graph where edges represent influence based on publication windows (X=5/10 years). Calculated **In-Degree** and **Out-Degree** for authors. |

---

## Execution Instructions

### Hadoop MapReduce
1. Move the dataset to HDFS:
   ```bash
   hdfs dfs -put ./Dataset/200.txt /input_dir/

Run the Job:

Bash
hadoop jar Hadoop_Code/WordCount.jar WordCount /input_dir /output_dir
Apache Spark
Ensure the PySpark environment is active.

Run the analysis scripts:

Bash
python3 Spark_Code/author_network.py

Key Observations
Scalability: Spark's DataFrames proved significantly faster than RDDs for the self-join operations in the Author Influence Network.

Data Locality: Adjusting split sizes in Hadoop directly impacted the number of Map tasks and overall execution time.

Vectorization: TF-IDF vectors provided a robust way to quantify document similarity, though pairwise cosine similarity is computationally expensive for large datasets.