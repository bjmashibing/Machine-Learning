package com.msb.traffic

import java.text.SimpleDateFormat
import java.util.Calendar

import net.sf.json.JSONObject
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, HasOffsetRanges, KafkaUtils, OffsetRange}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
    /**
      * 将每个卡扣的总速度_车辆数  存入redis中
      * 【yyyyMMdd_Monitor_id,HHmm,SpeedTotal_CarCount】
      */
    object CarEventCountAnalytics {

      def main(args: Array[String]): Unit = {
      // Create a StreamingContext with the given master URL
      val conf = new SparkConf().setAppName("CarEventCountAnalytics")
      conf.setMaster("local[*]")
      val ssc = new StreamingContext(conf, Seconds(5))
      //    ssc.sparkContext.setCheckpointDir("hdfs://node01:9000/checkpoint/ss1")
      // Kafka configurations
      val topics = Set("car_events")
      //sparkstreaming+kafka direct方式   recevier方式
      val brokers = "node01:9092,node02:9092,node03:9092"

      val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> brokers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "predictGroup",//
//      "auto.offset.reset" -> "earliest",
      "enable.auto.commit" -> (false: java.lang.Boolean)//默认是true
    )

    val dbIndex = 10
    // Create a direct stream
    val kafkaStream: InputDStream[ConsumerRecord[String, String]] = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    val events: DStream[JSONObject] = kafkaStream.map(line => {
      //JSONObject.fromObject 将string 转换成jsonObject
      val data: JSONObject = JSONObject.fromObject(line.value())
      println(data)
      data
    })

    /**
     * carSpeed  K:monitor_id
     * 					 V:(speedCount,carCount)
      *
     */
    val carSpeed : DStream[(String, (Int, Int))]=
      events.map(jb => (jb.getString("camera_id"),jb.getInt("speed")))
        //(speed:Int)=>(speed,1) 给每辆车计数为1
        .mapValues((speed:Int)=>(speed,1))
//      (camera_id,(speed,1))
        .reduceByKeyAndWindow((a:Tuple2[Int,Int], b:Tuple2[Int,Int]) => {(a._1 + b._1, a._2 + b._2)},Seconds(60),Seconds(10))
//        .reduceByKeyAndWindow((a:Tuple2[Int,Int], b:Tuple2[Int,Int]) => {(a._1 + b._1, a._2 + b._2)},(a:Tuple2[Int,Int], b:Tuple2[Int,Int]) => {(a._1 - b._1, a._2 - b._2)},Seconds(20),Seconds(10))

    //将计算结果保存到redis中
    carSpeed.foreachRDD(rdd => {
      rdd.foreachPartition(partitionOfRecords => {
        val jedis = RedisClient.pool.getResource
        partitionOfRecords.foreach(pair => {
//          (卡口号，（总车速，总的车辆数）)
          val camera_id = pair._1
          val speedTotal = pair._2._1
          val CarCount = pair._2._2
          val now = Calendar.getInstance().getTime()
          // create the date/time formatters
          val dayFormat = new SimpleDateFormat("yyyyMMdd")
          val minuteFormat = new SimpleDateFormat("HHmm")
          val day = dayFormat.format(now)  //20200220
          val time = minuteFormat.format(now) //2046
          if(CarCount!=0&&speedTotal!=0){
            jedis.select(dbIndex)
            //20190928_310999002103   -- k,v  -- (1607,200_5)
//            hset  hash类型（key（key，value））
            jedis.hset(day + "_" + camera_id, time , speedTotal + "_" + CarCount)
          }
        })
        RedisClient.pool.returnResource(jedis)
      })
    })

    /**
      * 异步更新offset
      * kafkaStream.foreachRDD { rdd =>
      * val offsetRanges: Array[OffsetRange] = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
      * // some time later, after outputs have completed
      *       kafkaStream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
      * }
      */
    ssc.start()
    ssc.awaitTermination()
  }
}