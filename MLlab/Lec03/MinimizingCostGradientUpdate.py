import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# 텐서플로우 베리어블로 선언하고 랜던값을 넣음
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis
hypothesis = X * W

# cost function
cost = tf.reduce_mean(tf.reduce_sum(tf.square(hypothesis - Y)))

# Minimize
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# 세션 열고 변수 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    # W값을 업데이트하게 실행을 시켜 X와 Y값을 던져주고
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    # 잘 업데이트가 됐는지 cost와 W 값을 print로 출력해줍니다
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))




# 이렇게 미니마이즈를 수동으로 설정해주었지만 텐서플로우에서는 편한 라이브러리인 그레디언트디센트옵티마이저
# 가 있어 더 쉽게 만들 수 있다.