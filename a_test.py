import tensorflow as tf 

def foo():
  with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
    v = tf.compat.v1.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2