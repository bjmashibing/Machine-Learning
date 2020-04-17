package com.msb.kmeans

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer

import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.feature.IDFModel
import org.apache.spark.rdd.RDD
import org.wltea.analyzer.lucene.IKAnalyzer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.KMeansModel

object BlogKMeans {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("BlogKMeans").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val rdd = sc.textFile("data/original.txt",8)

//    map mapPartition  HBase Redis ES      foreachPartition   foreach
    /**
      * mapPartitions 将每一篇微博内容进行分词 IKAnalyzer分词器（过滤掉停用词）
      * IKAnalyzer jar分发到每一台worker上      NM
      *
      *
      * wordRDD：
      *   k:微博ID
      *   V:Array(word1,word2.....)
      */
    var wordRDD: RDD[(String, ArrayBuffer[String])] = rdd.mapPartitions(iterator => {
      val list = new ListBuffer[(String, ArrayBuffer[String])]
      while (iterator.hasNext) {
        //创建分词对象   IKAnalyzer支持两种分词模式：最细粒度和智能分词模式，如果构造函数参数为false，那么使用最细粒度分词。
        val analyzer = new IKAnalyzer(true)
        val line = iterator.next()
        val textArr = line.split("\t")
        val id = textArr(0)
        val text = textArr(1)
        //分词     第一个参数只是标识性，没有实际作用，第二个读取的数据
        val ts: TokenStream = analyzer.tokenStream("", text)
        //得到相应词汇的内容
        val term: CharTermAttribute = ts.getAttribute(classOf[CharTermAttribute])
        //重置分词器，使得tokenstream可以重新返回各个分词
        ts.reset()
        val arr = new ArrayBuffer[String]
        //遍历分词数据
        while (ts.incrementToken()) {
          arr.+=(term.toString())
        }

        list.append((id, arr))
        analyzer.close()
      }
      list.iterator
    })

    /**
      * action算子之后，才会将wordRDD 写入到内存中
      * application  1job  cache没意思
      * persist()
      */
    wordRDD = wordRDD.cache()
//    val allDicCount = wordRDD.flatMap(one=>{one._2}).distinct().count()
    /**
      * 1000:生成长度为1000的向量
      *
      * Hash碰撞的问题
      */
    val hashingTF: HashingTF = new HashingTF(1000)

    /**
      * 为了提高训练模型的效率，向量长度（特征个数）设置为1000个
      *
      * 向量不同的位置的值与每篇微博的单词并不是一一对应的
      */
    val tfRDD: RDD[(String, Vector)] = wordRDD.map(x => {
      (x._1, hashingTF.transform(x._2))
    })


  /**
      * tfRDD
      * K:微博ID
      * V:Vector（tf，tf，tf.....）
      *
      * hashingTF.transform(x._2)
      * 按照hashingTF规则 计算分词频数（TF）
      * IDF
      */
    val idf: IDFModel = new IDF().fit(tfRDD.map(_._2))

    /**
     * K:微博 ID
     * V:每一个单词的TF-IDF值
     * tfIdfs这个RDD中的Vector就是训练模型的训练集
     * 计算TFIDF值
     */

    val tfIdfs: RDD[(String, Vector)] = tfRDD.mapValues(idf.transform(_))
//    tfIdfs.foreach(println)

    //设置聚类个数
    val kcluster = 20
    val kmeans = new KMeans()
    kmeans.setK(kcluster)

    //使用的是kemans++算法来训练模型  "random"|"k-means||"  kmeans kmeans++
    kmeans.setInitializationMode("k-means||")
    //设置最大迭代次数  本次中心的坐标
    kmeans.setMaxIterations(1000)
    //训练模型   聚类的模型(k个中心点的坐标)      线性回归（w0,w1）
    val kmeansModel: KMeansModel= kmeans.run(tfIdfs.map(_._2))
    //    kmeansModel.save(sc, "d:/model001")
    //打印模型的20个中心点
    println(kmeansModel.clusterCenters)

    /**
     * 模型预测
     */
    val modelBroadcast = sc.broadcast(kmeansModel)
    /**
      * predicetionRDD KV格式的RDD
      * 	K：微博ID
      * 	V：分类号
      */
    val predicetionRDD: RDD[(String, Int)] = tfIdfs.mapValues(vetor => {
      val model = modelBroadcast.value
      model.predict(vetor)
    })
    /**
      * 总结预测结果
      * tfIdfs2wordsRDD:kv格式的RDD
      * K：微博ID
      * V：二元组(Vector(tfidf1,tfidf2....),ArrayBuffer(word,word,word....))
      */
    val tfIdfs2wordsRDD: RDD[(String, (Vector, ArrayBuffer[String]))] = tfIdfs.join(wordRDD)

    /**
     * result:KV
     * K：微博ID
     * V:(类别号，(Vector(tfidf1,tfidf2....),ArrayBuffer(word,word,word....)))
     */
    val result : RDD[(String, (Int, (Vector, ArrayBuffer[String])))] =
          predicetionRDD.join(tfIdfs2wordsRDD)
    /**
     * 查看1号类别中tf-idf比较高的单词，能代表这类的主题
     */
    result
      .filter(x => x._2._1 == 2)
      .flatMap(line => {

      val tfIdfV: Vector = line._2._2._1
      val words: ArrayBuffer[String] = line._2._2._2
      val list = new ListBuffer[(Double, String)]

      for (i <- 0 until words.length) {
        //hashingTF.indexOf(words(i)) 当前单词在1000个向量中的索引号

        list.append((tfIdfV(hashingTF.indexOf(words(i))), words(i)))
      }
      list
    })
      .sortBy(x => x._1, false)
      .map(_._2).distinct()
      .take(30).foreach(println)

    sc.stop()
  }
}
