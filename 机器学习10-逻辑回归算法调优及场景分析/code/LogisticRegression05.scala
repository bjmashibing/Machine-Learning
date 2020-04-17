package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

//鲁棒性调优 提高模型抗干扰能力
object LogisticRegression05 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate()
    val data = spark.read.format("libsvm")
      .load("data/健康状况训练集.txt")
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0), splits(1))

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.2)
      /**
        * 0:Coefficients: [0.03848957195367213,0.14631693503425972,0.0054828600613679905,-0.004313250356126902,-6.182036099035085E-4,-0.011291909066241251] Intercept: -1.5413240962907293
        * 1:Coefficients: (6,[0],[0.006998233007240396]) Intercept: -0.29467948397444715
        */
      .setElasticNetParam(1)
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

    println("正确率：" + (1-(mean/testData.count())))
    println("正确率：" + lrModel.evaluate(testData).accuracy)
    spark.close()
  }
}
