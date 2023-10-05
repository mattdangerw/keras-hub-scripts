import tensorflow as tf

@tf.function(jit_compile=True)
def f(x, y):
    @tf.autograph.experimental.do_not_convert
    def foo(x, y):
        return {"foo": x ** 2 + y, "bar": x ** 2 - y}
    return foo(x, y)

x = tf.constant([2, 3])
y = tf.constant([3, -2])
f(x, y)
f(x, y)
