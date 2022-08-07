# Lab 4 Multi-variable linear regression
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# None는 현재 인스턴스의 갯수가 5개지만 추후에 추가 할 수도 있어서 일단은 임의의 값 None로 설정하였다. X의 쉐입에서 뒤에 3은 variable의 갯수를 표현하였다.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Multi Variable Linear Regression의 Hypothesis 식이다.
hypothesis = tf.matmul(X, W) + b

# Linear regression의 cost function이다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘의 식이고 Learning Rate값은 1e-5로 주겠다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
