package com.msb.traffic

import java.text.SimpleDateFormat
import java.util
import java.util.Date

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
/**
 * 训练模型
 */
object TrainLRwithLBFGS {

    val sparkConf = new SparkConf().setAppName("train traffic model").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    // create the date/time formatters
    val dayFormat = new SimpleDateFormat("yyyyMMdd")
    val minuteFormat = new SimpleDateFormat("HHmm")

    def main(args: Array[String]) {
        // fetch data from redis
        val jedis = RedisClient.pool.getResource
        jedis.select(10)

        /**
          * 分别训练310999003001、310999003001的模型
          */
        val camera_ids = List("310999003001","310999003102")

        /**
          * 310999003001、310999003102这两条路相邻路的卡口号
          */
        val camera_relations:Map[String,Array[String]] = Map[String,Array[String]](
            "310999003001" -> Array("310999003001","310999003102","310999000106","310999000205","310999007204"),
            "310999003102" -> Array("310999003001","310999003102","310999000106","310999000205","310999007204")
        )
        val temp = camera_ids.map({ camera_id =>
            //拿最近5h的数据作为训练集
            val hours = 5
            //当前毫秒数据
            val nowtimelong = System.currentTimeMillis()
            val now = new Date(nowtimelong)
            val day = dayFormat.format(now)//yyyyMMdd

            //获取310999003001相邻路段的卡口号
            val array = camera_relations.get(camera_id).get

            /**
             * relations中存储了每一个卡扣在day这一天每一分钟的平均速度
             */
            val relations: Array[(String, util.Map[String, String])]  = array.map({ camera_id =>
//                println(camera_id)
                // minute_speed_car_map保存的每分钟 的路况情况
                val minute_speed_car_map: util.Map[String, String] = jedis.hgetAll(day + "_'" + camera_id + "'")
                (camera_id, minute_speed_car_map)
            })

            // 保存训练集数据
            val dataSet = ArrayBuffer[LabeledPoint]()
            //Range 从300到1 递减 不包含0
            for(i <- Range(60*hours,0,-1)){

                val features = ArrayBuffer[Double]()
                val labels = ArrayBuffer[Double]()
                //[0,1,2]
                for(index <- 0 to 2){
                    //当前时刻过去的时间那一分钟  i:300-0=300
                    /**
                      * 21：40   16:42
                      */
                    val tempOne = nowtimelong - 60 * 1000 * (i-index)
                    val d = new Date(tempOne)
                    val tempMinute = minuteFormat.format(d)//HHmm  1640
                    //+1分钟  16:43
                    val tempNext = tempOne - 60 * 1000 * (-1)
                    val dNext = new Date(tempNext)
                    val tempMinuteNext = minuteFormat.format(dNext)//HHmm

                    for((k,v) <- relations){
                        val map = v //map -- k:HHmm    v:Speed_count
                        if(index == 2 && k == camera_id){
                            // 获取当前路段16：43分钟的平均速度
                            if (map.containsKey(tempMinuteNext)) {
                                val info = map.get(tempMinuteNext).split("_")
                                val f = info(0).toFloat / info(1).toFloat
                                labels += f
                            }
                        }
                        //tempMinute：1642
                        if (map.containsKey(tempMinute)){
                            val info = map.get(tempMinute).split("_")
                            val f = info(0).toFloat / info(1).toFloat
                            features += f
                        } else{
                            features += -1.0
                        }
                    }
                }
                //LabeledPoint封装的就是一条训练集数据 （x,y）
                if(labels.toArray.length == 1 ){
                    //array.head 返回数组第一个元素
                    val label = (labels.toArray).head
                    val record =
                        //if ((label.toInt/10)<10) (label.toInt/10) else 10.0 计算结果0-10 11中可能
                        LabeledPoint(if ((label.toInt/10)<10) (label.toInt/10) else 10.0, Vectors.dense(features.toArray))
                    dataSet += record
                }
            }

          val data: RDD[LabeledPoint] = sc.parallelize(dataSet)

            // Split data into training (80%) and test (20%).
            //将data这个RDD随机分成 8:2两个RDD
            val splits = data.randomSplit(Array(0.8, 0.2))
            //构建训练集
            val training = splits(0)
            /**
             * 测试集的重要性：
             * 	测试模型的准确度，防止模型出现过拟合的问题
             */
            val test = splits(1)

            if(!data.isEmpty()){
                // 训练逻辑回归模型LogisticRegressionWithLBFGS 可以给我训练出来一个具备多分类功能的模型
                val model = new LogisticRegressionWithLBFGS()
                  //11分类
                        .setNumClasses(11)
                        .setIntercept(true)
                        .run(training)
                // 测试集测试模型
                val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
                    val prediction = model.predict(features)
                    (prediction, label)
                }

                predictionAndLabels.foreach(x=> println("预测类别："+x._1+",真实类别："+x._2))

                // Get evaluation metrics. 得到评价指标
                val metrics: MulticlassMetrics = new MulticlassMetrics(predictionAndLabels)
                val precision = metrics.accuracy// 准确率
                println("Precision = " + precision)

                /**
                  * 如果模型的准确率超过80%  模型保存在hdfs上
                  */
                if(precision > 0.8){

                    val path = "hdfs://node01:9000/model/model_"+camera_id+"_"+nowtimelong
                    model.save(sc, path)
                    println("saved model to "+ path)
                    //不同路段对应模型路径保存在redis中
                    jedis.hset("model", camera_id , path)
                }
            }
        })
        RedisClient.pool.returnResource(jedis)
    }
}
