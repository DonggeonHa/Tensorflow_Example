import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0)

# hypothesis
hypothesis = X * W

# cost function
cost = tf.reduce_mean(tf.reduce_sum(tf.square(hypothesis - Y)))

# Minimize
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# 세션 열고 변수 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
   print(step, sess.run(W))
   sess.run(train)

