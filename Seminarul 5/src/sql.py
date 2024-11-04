#!/usr/bin/env python3
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import count, desc, avg


def main():
    spark = SparkSession.builder.appName("MyApp").getOrCreate()

    dataframe = read(spark)

    load_dataframe_as_table(dataframe)
    # debug_data(dataframe)

    query_cashiers(spark)
    query_cities(spark)
    query_jobs(spark)
    query_distinct_jobs(spark)


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


def load_dataframe_as_table(dataframe: DataFrame):
    dataframe.createTempView("employees")


def query_cashiers(spark: SparkSession):
    """
    Task 1: Identify the three employees with the "Cashier" job who have the highest salaries. Show
            their last and first names and their salaries
    """
    result = spark.sql(
        """
        SELECT e.lname, e.fname
        FROM employees e
        WHERE e.job_title = 'Cashier'
        ORDER BY e.salary DESC
        LIMIT 3;
        """
    )
    result.show()


def query_cities(spark: SparkSession):
    """
    Task 2: Which is the city with the highest number of employees?
    """
    result = spark.sql(
        """
        SELECT e.city, count(e.city) AS count
        FROM employees e
        GROUP BY e.city
        ORDER BY count DESC
        LIMIT 1;
        """
    )
    result.show()


def query_jobs(spark: SparkSession):
    """
    Task 3: For each job_title show the number of employees and the average salary
    """
    result = spark.sql(
        """
        SELECT e.job_title, count(e.job_title) AS employee_count, avg(e.salary) AS employee_avg_salary
        FROM employees e
        GROUP BY e.job_title;
        """
    )
    result.show()


def query_distinct_jobs(spark: SparkSession):
    """
    Task 4: Count how many distinct job_titles there are.
    """
    result = spark.sql(
        """
        SELECT count(x.job_title) AS distinct_titles
        FROM (
            SELECT DISTINCT e.job_title
            FROM employees e
        ) AS x;
        """
    )
    result.show()


if __name__ == "__main__":
    main()
