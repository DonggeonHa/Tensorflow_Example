import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.)

# hypothesis
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost function
cost = tf.reduce_mean(tf.reduce_sum(tf.square(hypothesis - Y)))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# gradients를 계산한 값을 돌려준다.
gvs = optimizer.compute_gradients(cost, [W])

# gradients값을 더하거나 빼는 등 임의로 조정하면 된다

# 수정한 gvs 값을 다시 optimizer에 넣는다.
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
