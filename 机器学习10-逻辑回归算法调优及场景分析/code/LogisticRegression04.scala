package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  *正负样本采样失衡
  * 上采样:少量类别的样本增加
  * 下采样:大量类别的样本减少
  */

object LogisticRegression04 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate()
    val data = spark.read.format("libsvm")
      .load("data/健康状况训练集.txt")

    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0), splits(1))

    //数据发生了严重的失衡    癌症样本只有1条   非癌症样本
//    val training = trainingData.filter(col("label")===1).union(trainingData.filter(col("label")===0).limit(1))
    /**
      * 样本均衡
      *   正确率：0.7536182530223055
      * 样本失衡
      *   正确率：0.5063851523923038
      */
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setFitIntercept(true)
    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    //测试集验证正确率
    val testRest = lrModel.transform(testData)
    //打印结果
    testRest.show(false)
    testRest.printSchema()

    //计算正确率
    val mean = testRest.rdd.map(row => {
      val label = row.getAs[Double]("label")
      val prediction = row.getAs[Double]("prediction")
      math.abs(label - prediction)
    }).sum()

    println("正确率：" + (1-(mean/testData.count())))
    println("正确率：" + lrModel.evaluate(testData).accuracy)
    spark.close()
  }
}
