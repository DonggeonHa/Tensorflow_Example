# Lab 5 Logistic Regression Classifier
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)  # for reproducibility

# 멀티 베리어블 데이터입니다.
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
# 바이너리 클래스피케이션이라 0 또는 1로 설정
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# 쉐입은 항상 데이터를 보고 결정해야하는데 X의 경우 x_data의 베리어블 갯수가 2여서 2를 입력 Y는 1을 넣음
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 쉽게 들어오는 값의 갯수가 앞에 두고, 나가는값의 갯수를 뒤에 두면 된다.
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
# bias의 경우 항상 나가는 값의 갯수와 같다. 즉 Y의 갯수가 1이니 1로 선언.
b = tf.Variable(tf.random_normal([1]), name='bias')

# Logistic Classification에서 정의한 Hypothesis 이다.
# WX + b 부분이 e의 -WX 승과 같다
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Logistic Classification에서 정의한 cost/loss function 이다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 예측한 값을 가지고 테스트
# Accuracy computation
# 0.5를 기준으로 True인지 False인지 계산 한다
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 그 다음 예측한 값들이 얼마나 정확한지, 실제 데이터 Y와 같은지 비교해서 True면 1 False는 0으로 해서 계산하고 그 값을 평균을 내게되면 accuracy가 나온다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    # 학습된 모델을 가지고 가설, predicted, accuracy를 구한다.
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496

Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''