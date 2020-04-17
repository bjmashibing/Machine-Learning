package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

//线性不可分 ----升高维度
object LogisticRegression03 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate()
    import spark.implicits._
    val data = spark.read.format("libsvm")
      .load("data/线性不可分数据集.txt")

      //升高维度
      .rdd
      .map(row => {
        val label = row.getAs[Double]("label")
        val vector = row.getAs[SparseVector]("features").toDense
        val newFs = new DenseVector(Array(vector(0), vector(1), vector(0) * vector(1)))
        (label, newFs)
      }).toDF("label", "features")
    //将数据切割训练集和测试集
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0), splits(1))

    val lr = new LogisticRegression()
      .setMaxIter(10)
      //是否有截距
      .setFitIntercept(true)
    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    //测试集验证正确率
    val testRest = lrModel.transform(testData)
    //打印结果
    testRest.show(false)
    //计算正确率
    val mean = testRest.rdd.map(row => {
      val label = row.getAs[Double]("label")
      val prediction = row.getAs[Double]("prediction")
      math.abs(label - prediction)
    }).sum()

    println("正确率：" + (1 - (mean / testData.count())))
    println("正确率：" + lrModel.evaluate(testData).accuracy)
    spark.close()
  }
}
