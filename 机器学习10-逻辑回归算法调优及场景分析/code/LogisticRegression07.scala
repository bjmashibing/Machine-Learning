package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.SparkSession

//最大最小值归一化
object LogisticRegression07 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate()
    val data = spark.read.format("libsvm")
      .load("data/环境分类数据.txt")

    val minMaxScalerModel = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("minmaxFeatures")
      .fit(data)
    val features = minMaxScalerModel.transform(data)
    features.show(10)

    val splits = features.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0), splits(1))

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setFeaturesCol("minmaxFeatures")

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
