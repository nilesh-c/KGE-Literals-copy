package net.sansa_stack.ml.kge

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}

/**
  * Created by nilesh on 31/05/2017.
  */
object MaxMarginLoss {
  def apply(margin: Float): (Symbol, Symbol) => Symbol = {
    loss(margin) _
  }

  def loss(margin: Float)(positiveScore: Symbol, negativeScore: Symbol): Symbol = {
    var loss = s.max(negativeScore - positiveScore + margin, 0)
    loss = s.sum(name = "sum")()(Map("data" -> loss))
    s.make_loss(name = "loss")()(Map("data" -> loss))
  }
}

object Sigmoid {
  def apply(x: Symbol): Symbol = {
    s.Activation(name = "sigmoid")()(Map("data" -> x, "act_type" -> "sigmoid"))
  }
}

object Tanh {
  def apply(x: Symbol): Symbol = {
    s.Activation(name = "tanh")()(Map("data" -> x, "act_type" -> "tanh"))
  }
}

object L2Similarity {
  def apply(x: Symbol, y: Symbol): Symbol = {
    val difference = x - y
    var score = s.square()()(Map("data" -> difference))
    score = s.sum()()(Map("data" -> score, "axis" -> 0))
    score*(-1.0)
  }
}

object DotSimilarity {
  def apply(x: Symbol, y: Symbol): Symbol = {
    s.dot("dot")()(Map("lhs" -> x, "rhs" -> y))
  }
}

object Hits {
  def hitsAt1(label: NDArray, predicted: NDArray): Float = {
    val labelA = label.toArray
    val predA = predicted.toArray
    labelA.zip(predA).map(x => if(x._1.toInt == x._2.toInt && x._1.toInt == 1) 1 else 0).sum
  }
}