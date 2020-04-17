package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession


object LogisticRegression01 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate()
    import spark.implicits._

    val dataRDD = spark.sparkContext.textFile("data/breast_cancer.csv")
    val data = dataRDD.map(x=>{
      val arr = x.split(",")
      val features = new Array[String](arr.length-1)
      arr.copyToArray(features,0,arr.length-1)
      val label = arr(arr.length - 1)
      (new DenseVector(features.map(_.toDouble)),label.toDouble)
    }).toDF("features","label")


    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0), splits(1))


    val lr = new LogisticRegression()
      .setMaxIter(10)
    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


    //    lrModel.setThreshold(0.3)


    //测试集验证正确率
    val testRest = lrModel.transform(testData)
    //打印结果
    testRest.show(false)
    //计算正确率
    val mean = testRest.rdd.map(row => {
      //这个样本真实的分类号
      val label = row.getAs[Double]("label")
      //将测试数据的x特征带入到model后预测出来的分类号
      val prediction = row.getAs[Double]("prediction")
      //0:预测正确   1:预测错了
      math.abs(label - prediction)
    }).sum()

    println("正确率：" + (1-(mean/testData.count())))

    println("正确率：" + lrModel.evaluate(testData).accuracy)

  //在特定场合  要自定义分类阈值
    val count = testRest.rdd.map(row => {
      val probability = row.getAs[DenseVector]("probability")
      val label = row.getAs[Double]("label")
      val score = probability(1)
      val prediction = if(score > 0.3) 1 else 0
      math.abs(label - prediction)
    }).sum()
    println("自定义分类阈值 正确率：" + (1 - (count / testData.count())))


    spark.close()
  }
}
