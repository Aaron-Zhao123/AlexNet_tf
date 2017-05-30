import tensorflow as tf
import numpy as np


val_a = np.array([[-1,0,1,2,3,4,5],[-1,0,1,2,3,4,5]])
target = tf.constant(4,tf.int32)
a = tf.constant(val_a ,tf.int32)

tmp = tf.equal(a, target)
exist = tf.reduce_sum(tf.cast(tmp,tf.float32),axis=1)

reordered = tf.nn.top_k(a, k=3).indices

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(reordered.eval())
    print(exist.eval())
    print(tmp.eval())
