package wordcount

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.{SortedMap, mutable}

object Program {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession
      .builder()
      .appName("Java Spark SQL")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = sparkSession.sparkContext

    val file = sc.textFile("hdfs://localhost:9000/user/root/input/man.txt")
    val stopWords = new mutable.HashSet() ++ new StopWordsRemover().getStopWords

    val result = file.flatMap(line => line.split("[,.!?:;\\s]"))
      .filter(word => word.length >= 2 && !stopWords.contains(word))
      .map(word => (word.toLowerCase, 1))
      .reduceByKey(_ + _)

    val dataFrame = sparkSession
      .createDataFrame(result)
      .toDF("word", "count")
      .sort(desc("count"))
      .limit(100)

    dataFrame
      .write
      .csv("result")
  }
}
