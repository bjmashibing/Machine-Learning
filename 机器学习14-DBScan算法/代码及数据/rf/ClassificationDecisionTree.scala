package com.msb.rf

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/**
 * 决策树
 */
object ClassificationDecisionTree {

  def main(args: Array[String]): Unit = {
	  val conf = new SparkConf()
			  conf.setAppName("ClassificationDecisionTree")
			  conf.setMaster("local[3]")
		val sc = new SparkContext(conf)
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/汽车数据样本.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    //指明分类的类别
    val numClasses=2
    //指定离散变量，未指明的都当作连续变量处理
    //某列下有1,2,3类别 处理时候要自定为4类，虽然没有0，但是程序默认从0开始分类
    //这里天气维度有3类,但是要指明4,这里是个坑,后面以此类推
    val categoricalFeaturesInfo=Map[Int,Int](0->4,1->4,2->3,3->3)
    //设定评判标准  "gini"/"entropy"
    val impurity="entropy"//信息熵来表示信息混乱程度  国家收入差异
    //树的最大深度,太深运算量大也没有必要  剪枝   防止模型的过拟合！！！
    val maxDepth=3
    //设置离散化程度,连续数据需要离散化,分成32个区间,默认其实就是32,分割的区间保证数量差不多  这个参数也可以进行剪枝
    val maxBins=32
    //生成模型
    val model =DecisionTree.trainClassifier(trainingData,numClasses,categoricalFeaturesInfo,
                    impurity,maxDepth,maxBins)

    val labelAndPreds: RDD[(Double, Double)] = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    //测试
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    //打印出决策树
    println("Learned classification tree model:\n" + model.toDebugString)
  }
}
