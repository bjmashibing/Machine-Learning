package com.msb.lr_new

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

object VectorTest {
  def main(args: Array[String]): Unit = {
    //稀疏向量
    val sparseVector = new SparseVector(6,Array(1,3,5),Array(1,1,1))
    println(sparseVector)
    val denseVector = new DenseVector(Array(1,1,0,1))
    println(denseVector)



  }
}
