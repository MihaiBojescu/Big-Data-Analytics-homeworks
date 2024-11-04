#!/usr/bin/env python3
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import count, desc, avg


def main():
    spark = SparkSession.builder.appName("MyApp").getOrCreate()

    dataframe = read(spark)

    # debug_data(dataframe)

    query_cashiers(dataframe)
    query_cities(dataframe)
    query_jobs(dataframe)
    query_distinct_jobs(dataframe)


def read(spark: SparkSession):
    schema = StructType(
        [
            StructField("emp_id", StringType(), False),
            StructField("fname", StringType(), False),
            StructField("lname", StringType(), False),
            StructField("address", StringType(), False),
            StructField("city", StringType(), False),
            StructField("state", StringType(), False),
            StructField("zipcode", StringType(), False),
            StructField("job_title", StringType(), False),
            StructField("email", StringType(), False),
            StructField("active", StringType(), False),
            StructField("salary", IntegerType(), False),
        ]
    )

    employees_rdd = (
        spark.read.option("mergeSchema", "true")
        .option("delimiter", "\t")
        .csv("./data/employees", inferSchema=True, header=False)
    )
    employees_df = spark.createDataFrame(employees_rdd.rdd, schema)

    return employees_df


def debug_data(dataframe: DataFrame):
    dataframe.printSchema()
    dataframe.show()


def query_cashiers(dataframe: DataFrame):
    """
    Task 1: Identify the three employees with the "Cashier" job who have the highest salaries. Show
            their last and first names and their salaries
    """
    result = (
        dataframe.select(dataframe.lname, dataframe.fname)
        .where(dataframe.job_title == "Cashier")
        .orderBy(desc(dataframe.salary))
        .limit(3)
    )
    result.show()


def query_cities(dataframe: DataFrame):
    """
    Task 2: Which is the city with the highest number of employees?
    """
    result = (
        dataframe.select(dataframe.city)
        .groupBy(dataframe.city)
        .count()
        .sort(desc("count"))
        .limit(1)
    )
    result.show()


def query_jobs(dataframe: DataFrame):
    """
    Task 3: For each job_title show the number of employees and the average salary
    """
    result = (
        dataframe.select(dataframe.job_title, dataframe.salary)
        .groupBy(dataframe.job_title)
        .agg(
            count(dataframe.job_title).alias("employee_count"),
            avg(dataframe.salary).alias("employee_avg_salary"),
        )
    )
    result.show()


def query_distinct_jobs(dataframe: DataFrame):
    """
    Task 4: Count how many distinct job_titles there are.
    """
    result = dataframe.select(dataframe.job_title).distinct().count()
    print(f"Number of distinct jobs: {result}")


if __name__ == "__main__":
    main()
