import tensorflow as tf
from typing import Tuple

@tf.function
def calc_prec_recall(batch: tf.Tensor, x_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    
    batch = tf.cast((batch > 0.5), tf.bool)
    x_hat = tf.cast((x_hat > 0.5), tf.bool)

    tp = tf.math.count_nonzero(tf.math.logical_and(batch, x_hat), dtype=tf.float32)
    tn = tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(x_hat),
                                                   tf.math.logical_not(batch)),
                               dtype=tf.float32)
    fp = tf.math.count_nonzero(tf.math.logical_and(x_hat, tf.math.logical_not(batch)), dtype=tf.float32)
    fn = tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(x_hat), batch), dtype=tf.float32)

    if (tf.math.add(tp, fp) != 0):
        prec = tf.math.divide(tp,  tf.math.add(tp, fp))
    else:
        prec = tf.constant(0., dtype=tf.float32)
    if (tf.math.add(tp, fn) != 0):
        recall = tf.math.divide(tp,  tf.math.add(tp, fn))
    else:
        recall = tf.constant(0., dtype=tf.float32)
    #f1 = 2 * precision * recall / (precision + recall)
    return (prec, recall)