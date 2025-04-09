# TF-IDF for Document Retrieval using PySpark MLLib

This project demonstrates how to perform document retrieval using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm with PySpark's MLLib library. It reads data from a tab-separated file, calculates TF-IDF scores, and identifies the document most relevant to a specific query term ("Gettysburg" in this case).

## Overview

The process involves the following steps:

1.  **Data Loading:** Reads text data from a specified file (e.g., `subset-small.tsv`) into a Spark Resilient Distributed Dataset (RDD).
2.  **Data Parsing:** Splits each line of the input file into fields and extracts the document content and document name.
3.  **Tokenization:** Splits the document content into individual words (tokens).
4.  **Term Frequency (TF) Calculation:** Uses `HashingTF` to convert each document (a sequence of words) into a sparse vector representing the term frequencies based on a hashing scheme.
5.  **Inverse Document Frequency (IDF) Calculation:** Computes the IDF for each term across the entire collection of documents. Terms that appear in many documents will have a lower IDF.
6.  **TF-IDF Calculation:** Multiplies the TF vector of each document by the IDF values to obtain the TF-IDF vector for each document. This vector represents the importance of each term within a document relative to the entire corpus.
7.  **Retrieval (Query):**
    * Hashes the query term ("Gettysburg") to find its corresponding hash value (index in the TF-IDF vectors).
    * Extracts the TF-IDF score for the query term's hash value from each document's TF-IDF vector.
    * Zips these scores with the corresponding document names.
    * Finds the document with the highest TF-IDF score for the query term, indicating the most relevant document.

## Prerequisites

* **Apache Spark:** Ensure you have Apache Spark installed and configured in your environment.
* **PySpark:** PySpark is the Python API for Spark and should be available with your Spark installation.

## Getting Started

1.  **Save the code:** Save the provided Python code as a `.py` file (e.g., `tfidf_retrieval.py`).
2.  **Prepare the data:** Place your input data file (e.g., `subset-small.tsv`) in the same directory as the Python script or provide the correct path to the file in the `sc.textFile()` function. The expected format is a tab-separated file where one of the columns contains the document text. The script assumes the document content is in the fourth column (index 3) and the document name is in the second column (index 1).
3.  **Run the script:** Execute the script using `spark-submit`:

    ```bash
    spark-submit tfidf_retrieval.py
    ```

## Code Explanation

```python
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
rawData = sc.textFile("subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])

# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  # 100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# At this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value.

# Let's compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

# I happen to know that the article for "Abraham Lincoln" is in our data
# set, so let's search for "Gettysburg" (Lincoln gave a famous speech there):

# First, let's figure out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Now we will extract the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# We'll zip in the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames)

# And, print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())