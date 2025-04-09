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