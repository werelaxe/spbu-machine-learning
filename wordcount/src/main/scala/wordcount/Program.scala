package wordcount

import org.apache.spark.sql.SparkSession

object Program {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession
      .builder()
      .appName("Java Spark SQL")
      .config("spark.master", "local")
      .getOrCreate()
      .sparkContext

    val file = sc.textFile("hdfs://localhost:9000/user/root/input/man.txt")
    val counts = file.flatMap(line => line.split(" "))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
    counts.foreach {
      println(_)
    }
  }
}
