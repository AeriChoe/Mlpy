import tensorflow as tf
import numpy as np
import os

# 경고 문구 없애기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 플레이스 홀더를 설정
X = tf.placeholder(tf.float32, shape=[None, 4]) # 값을 4개로 지정
Y = tf.placeholder(tf.float32, shape=[None, 1]) # 값을 1개로 지정

W = tf.Variable(tf.random_normal([4, 1]), name="weight") # 기본적 형태
b = tf.Variable(tf.random_normal([1]), name="bias") # 기본적 형태

# 가설을 설정하기
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# 사용자로부터 4가지 변수를 입력 받기
avg_temp = float(input('평균온도: '))
min_temp = float(input('최저온도: '))
max_temp = float(input('최고온도: '))
rain_fall = float(input('강수량: '))

# 입력한 변수로 예측한 결과값을 보여주기
with tf.Session() as sess:
    sess.run(model)
    save_path = "saved.cpkt"
    saver.restore(sess, save_path) # 저장된 값을 리스트로 가져오기

    # 입력 값을 이용하여 2차원 배열을 간단하게 만들어 보기
    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    # 입력한 데이터를 초기화
    arr = np.array(data, dtype=np.float32)
    # 실제로 예측하기
    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data}) # x에 담아서 돌려보기
    print(dict[0])
