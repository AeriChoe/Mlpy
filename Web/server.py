# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect
import datetime
import tensorflow as tf
import numpy as np
import os

# 경고 문구 없애기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

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

# 사용자의 요청이 있을 때마다 학습된 모델을 사용하기
sess = tf.Session()
sess.run(model)

# 저장된 모델을 세션에 적용
save_path = "model/saved.cpkt"
saver.restore(sess, save_path) # 저장된 값을 리스트로 가져오기

# render_template 이용해서 문서 보여주기
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp'])
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

    # 일단 배추가격 0으로 설정
    price = 0

    # 입력된 파라미터를 배열 형태로 준비
    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    # 입력한 데이터를 초기화
    arr = np.array(data, dtype=np.float32)
    # 실제로 예측하기
    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data}) # x에 담아서 돌려보기

    # price 안에 해당 결과 담길 수 있도록 하기
    price = dict[0]
    return render_template('index.html', price=price)

# 메인 함수로 정의 > flask 웹서버를 구동
if __name__ == '__main__':
   app.run(debug = True)
