from pyspark.sql import SparkSession
import sys

def main():
    # Check if input and output paths are provided
    if len(sys.argv) != 3:
        print("Usage: wordcount.py <input_path> <output_path>")
        sys.exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print("=============================================================")
    print("input_path: ", input_path)
    print("output_path: ", output_path)
    print("=============================================================")

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("WordCount") \
        .getOrCreate()
    
    print("=============================================================")
    print("Spark Session Created")
    print("=============================================================")


    # Read input file
    text_file = spark.read.text(input_path).rdd

    print("=============================================================")
    print("Text File Read")
    print("=============================================================")


    # Count words
    word_counts = (
        text_file.flatMap(lambda line: line.value.split())  # Split lines into words
        .map(lambda word: (word, 1))  # Create (word, 1) tuples
        .reduceByKey(lambda a, b: a + b)  # Reduce by key to sum word counts
    )

    print("=============================================================")
    print("Word count: ", word_counts)
    print("=============================================================")


    # Collect results and save to the output file
    # word_counts.saveAsTextFile(output_path)

    print("=============================================================")
    print("Word counts saved")
    print("=============================================================")


    # Stop the SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
