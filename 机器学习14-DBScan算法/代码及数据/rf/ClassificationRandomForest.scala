package com.msb.rf

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest
/**
 * 随机森林
 */
object ClassificationRandomForest {
  def main(args: Array[String]): Unit = {
	  val conf = new SparkConf()
			  conf.setAppName("analysItem")
			  conf.setMaster("local[3]")
		val sc = new SparkContext(conf)
    //读取数据
    val data =  MLUtils.loadLibSVMFile(sc,"data/汽车数据样本.txt")
    //将样本按7：3的比例分成
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    //分类数
    val numClasses = 2
    // categoricalFeaturesInfo 为空，意味着所有的特征为连续型变量
    val categoricalFeaturesInfo =Map[Int, Int](0->4,1->4,2->3,3->3)
    //树的个数
    val numTrees = 3
    //特征子集采样策略
    // 1：all 全部特征 。2：sqrt 把特征数量开根号后随机选择的 。 3：log2 取对数个。 4：onethird 三分之一
    val featureSubsetStrategy = "all"
    //纯度计算  "gini"/"entropy"
    val impurity = "entropy"
    //树的最大层次
    val maxDepth = 5
    //特征最大装箱数,即连续数据离散化的区间
    val maxBins = 32
    //训练随机森林分类器，trainClassifier 返回的是 RandomForestModel 对象
    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
        numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
//    //打印模型
//    println(model.toDebugString)
    //保存模型
   //model.save(sc,"汽车保险")
    //在测试集上进行测试
    val count = testData.map { point =>
        val prediction = model.predict(point.features)
    //    Math.abs(prediction-point.label)
        (prediction,point.label)
     }.filter(r => r._1 != r._2).count()
    println("Test Error = " + count.toDouble/testData.count().toDouble)
    println("model "+model.toDebugString)
  }
}