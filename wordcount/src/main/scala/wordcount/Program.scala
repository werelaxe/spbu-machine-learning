package wordcount

import org.apache.spark.sql.SparkSession

import scala.collection.SortedMap

object Program {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession
      .builder()
      .appName("Java Spark SQL")
      .config("spark.master", "local")
      .getOrCreate()
      .sparkContext

    val file = sc.textFile("hdfs://localhost:9000/user/root/input/man.txt")
    file.flatMap(line => line.split(" "))
      .filter(word => word.nonEmpty)
      .map(word => (word.toLowerCase, 1))
      .reduceByKey(_ + _)
      .sortBy(_._2, ascending = false)
      .take(100)
      .foreach { pair =>
        println(s"${pair._1}: ${pair._2}")
      }
  }
}
