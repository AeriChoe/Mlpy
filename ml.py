# 선형회귀란, 가중치가 곱해지는 형태로 식을 만듬.
import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import os

# 경고 문구 없애기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model initialization (모델을 초기화)
model = tf.global_variables_initializer()
# read the csv file
data = read_csv('price_data.csv', sep=',')

xy = np.array(data, dtype=np.float32)
#print(xy)

x_data = xy[:, 1:-1]
# print(x_data) <- 앞에서 부터 네개의 행의 데이터만 pick up
y_data = xy[:, [-1]]
# print(y_data) <- 가장 오른쪽 행의 데이터만 pick up

X = tf.placeholder(tf.float32, shape=[None, 4]) # 값을 4개로 지정
Y = tf.placeholder(tf.float32, shape=[None, 1]) # 값을 1개로 지정
W = tf.Variable(tf.random_normal([4, 1]), name="weight") # 기본적 형태
b = tf.Variable(tf.random_normal([1]), name="bias") # 기본적 형태

# 가설식(예측값)
hypothesis = tf.matmul(X, W) + b
# 비용함수(=손실함수_실제값과 예측값이 얼마나 차이나는지) 설정
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 최적화 함수 설정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005) # 매우 중요
train = optimizer.minimize(cost)
# 세션 값 만들기
sess = tf.Session()
# 글로벌 변수를 초기화
sess.run(tf.global_variables_initializer())

# 학습 진행 (10만회로 설정)
for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0: # 500회 진행 할때마다 print로 진행상황 확인
        print("#", step, "손실 비용:", cost_) # 코스트값 실시간으로 출력
        print("- 배추 가격:", hypo_[0]) # 첫번째 데이터에 대한 배추 가격 출력

# 학습 된 모델을 저장
saver = tf.train.Saver()
save_path = saver.save(sess, "saved.cpkt")
print("학습된 모델을 저장했습니다.")
