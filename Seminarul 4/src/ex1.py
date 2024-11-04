from pyspark.sql import SparkSession
from operator import add
from re import search, Match


def main():
    spark = SparkSession.builder.master("local").appName("word_count").getOrCreate()
    spark_context = spark.sparkContext
    file = spark_context.textFile("./data/shakespeare/comedies")

    counts = sorted(
        file.flatMap(lambda line: line.split(" "))
        .map(lambda word: safe_get(re.search("\\w+", word)))
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word, 1))
        .reduceByKey(add)
        .collect(),
        key=lambda entry: entry[1],
    )

    print(counts)


def safe_get(matches: Match | None):
    if not matches:
        return ""

    return matches[0]


if __name__ == "__main__":
    main()
