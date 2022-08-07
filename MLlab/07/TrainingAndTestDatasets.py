import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# 이 테스트 데이터 셋를 사용하여 모델 평가
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 정확한 예측 테스트 모델
prediction = tf.argmax(hypothesis, 1)                       # test data를 가지고 가설을 주고 예측을 한다면
is_correct = tf.equal(prediction, tf.argmax(Y, 1))          # 이것이 맞는지 아닌지를 측정
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  # is_correct 값을 평균 내서 정확성을 측정

# Launch graph
with tf.Session() as sess:
    # 세션을 열고 변수들 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # 테스트 데이터로 확인해보기
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
