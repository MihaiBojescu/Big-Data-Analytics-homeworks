from argparse import ArgumentParser
from math import sqrt
from sys import argv
from time import sleep
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def main():
    args_parser = ArgumentParser()
    args_parser.add_argument("--input-file", required=True, type=str)
    args_parser.add_argument("--output-folder", required=True, type=str)
    args = args_parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder

    spark = SparkSession.builder.appName("MyApp").getOrCreate()
    points = (
        spark.read.option("inferSchema", "true")
        .option("delimiter", " ")
        .csv(input_file)
        .toDF("x", "y", "centroid")
        .persist()
    )
    k = 3

    final_centroids = kmeans(spark, points, k)
    spark.createDataFrame(final_centroids, ["x", "y"]).write.csv(output_folder)

    input("Press any key to exit")
    spark.stop()


def kmeans(
    spark: SparkSession,
    data: DataFrame,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
):
    centroids = initialize_centroids(data, k)

    for i in range(max_iterations):
        print(f"Iteration {i}/{max_iterations}")

        assignments = assign_clusters(spark, data, centroids)
        new_centroids = update_centroids(assignments)

        movement = sum(
            calculate_distance((nc[0], nc[1]), c)
            for nc, c in zip(new_centroids, centroids)
        )

        centroids = new_centroids

        if movement < tolerance:
            break

    return centroids


def initialize_centroids(data: DataFrame, k: int) -> list:
    entries = []

    while len(entries) < k:
        rows = [
            (row["x"], row["y"])
            for row in data.sample(False, 1.0 / data.count()).limit(1).collect()
        ]
        entries.extend(rows)

    return entries


def assign_clusters(spark: SparkSession, data: DataFrame, centroids):
    centroids_broadcast = spark.sparkContext.broadcast(centroids)

    def assign_cluster(point: tuple[float, float]):
        min_distance = float("inf")
        nearest_centroid: tuple[float, float] = None

        for centroid in centroids_broadcast.value:
            distance = calculate_distance(point, centroid)

            if distance < min_distance:
                min_distance = distance
                nearest_centroid = centroid

        return nearest_centroid

    return data.rdd.map(
        lambda row: (assign_cluster((row["x"], row["y"])), [row["x"], row["y"]], 1)
    ).toDF(["centroid", "point", "count"])


def calculate_distance(a: tuple[float, float], b: tuple[float, float]):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def update_centroids(assignments: DataFrame):
    centroids_df = (
        assignments.withColumn("x", F.col("point").getItem(0))
        .withColumn("y", F.col("point").getItem(1))
        .groupBy("centroid")
        .agg(
            (F.sum("x") / F.count("x")).alias("x"),
            (F.sum("y") / F.count("y")).alias("y"),
        )
    )
    return [(row["x"], row["y"]) for row in centroids_df.collect()]


if __name__ == "__main__":
    main()
