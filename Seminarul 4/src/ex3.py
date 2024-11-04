from pyspark.sql import SparkSession
from operator import add, truediv
from re import search, Match


def main():
    spark = SparkSession.builder.master("local").appName("word_count").getOrCreate()
    spark_context = spark.sparkContext
    file = spark_context.textFile("./data/shakespeare/comedies")

    average_word_length = sorted(
        file.flatMap(lambda line: line.split(" "))
        .map(lambda word: safe_get(search("\\w+", word)))
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word[0], (1, len(word))))
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        .map(lambda x: (x[0], x[1][1] / x[1][0]))
        .saveAsTextFile("comedies.txt"),
        key=lambda entry: entry[1],
    )

    print(average_word_length)


def safe_get(matches: Match | None):
    if not matches:
        return ""

    return matches[0]


if __name__ == "__main__":
    main()
