package com.msb.traffic

import java.text.SimpleDateFormat
import java.util
import java.util.Date

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object PredictLRwithLBFGS {

  val sparkConf = new SparkConf().setAppName("predict traffic").setMaster("local[4]")
  val sc = new SparkContext(sparkConf)
  sc.setLogLevel("Error")
  // create the date/time formatters
  val dayFormat = new SimpleDateFormat("yyyyMMdd")
  val minuteFormat = new SimpleDateFormat("HHmm")
  val sdf = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss")

  def main(args: Array[String]) {

    val input = "2020-02-20_20:57:00"
    val date = sdf.parse(input)//yyyy-MM-dd_HH:mm:ss
    val inputTimeLong = date.getTime()
    val day = dayFormat.format(date)//yyyyMMdd

    // fetch data from redis
    val jedis = RedisClient.pool.getResource
    jedis.select(10)

    // find relative road monitors for specified road
    val camera_ids = List("310999003001", "310999003102")
    val camera_relations: Map[String, Array[String]] = Map[String, Array[String]](
      "310999003001" -> Array("310999003001", "310999003102", "310999000106", "310999000205", "310999007204"),
      "310999003102" -> Array("310999003001", "310999003102", "310999000106", "310999000205", "310999007204"))

    val temp = camera_ids.foreach({ camera_id =>
      val array = camera_relations.get(camera_id).get

      val relations: Array[(String, util.Map[String, String])] = array.map({ camera_id =>
        // fetch records of one camera for three hours ago
        (camera_id, jedis.hgetAll(day + "_'" + camera_id + "'"))
      })

      // organize above records per minute to train data set format (MLUtils.loadLibSVMFile)
      val featers = ArrayBuffer[Double]()
      // get current minute and recent two minutes
      for (index <- 3 to (1,-1)) {
        //拿到过去 一分钟，两分钟，过去三分钟的时间戳
        val tempOne = inputTimeLong - 60 * 1000 * index
        val currentOneTime = new Date(tempOne)
        //获取输入时间的 "HHmm"
        val tempMinute = minuteFormat.format(currentOneTime)//"HHmm"
//        println("inputtime ====="+currentOneTime)
        for ((k, v) <- relations) {
          val map = v //map : (HHmm,totalSpeed_total_carCount)
          if (map.containsKey(tempMinute)) {
            val info = map.get(tempMinute).split("_")
            val f = info(0).toFloat / info(1).toFloat
            featers += f
          } else {
            featers += -1.0
          }
        }
      }

      // Run training algorithm to build the model
      val path = jedis.hget("model", camera_id)
      if(path!=null){
        val model: LogisticRegressionModel = LogisticRegressionModel.load(sc, path)
        // Compute raw scores on the test set.
        val prediction = model.predict(Vectors.dense(featers.toArray))
        println(input + "\t" + camera_id + "\t" + prediction + "\t")
      }

    })

    RedisClient.pool.returnResource(jedis)
  }
}
