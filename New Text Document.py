import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def func(x1, x2, w):
    y1=x1*w[0]
    y2=x2*w[1]
    y3=y1+y2
    y4=max(y1, y2)
    y5=y3*w[2]+y4
    y6=y3-y4
    y7=1/(1+np.exp(-y5))
    y8=np.tanh(y6)
    y=y7+y8
    return y 

def get_data():
    x1=np.random.rand(10000)*10
    x2=np.random.rand(10000)*10
    w=[0.9, 0.4, 0.4]
    y=func(x1, x2, w)
    return x1, x2, y

x1=tf.compat.v1.placeholder(tf.float32)
x2=tf.compat.v1.placeholder(tf.float32)
y=tf.compat.v1.placeholder(tf.float32)

w1=tf.Variable(np.random.rand())
w2=tf.Variable(np.random.rand())
w3=tf.Variable(np.random.rand())
y1=tf.multiply(w1, x1)
y2=tf.multiply(w2, x2)
y3=tf.add(y1, y2)
y4=tf.maximum(y1, y2)
y5=tf.add(tf.multiply(y3, w3), y4)
y6=tf.subtract(y3, y4)
y7=tf.sigmoid(y5)
y8=tf.tanh(y6)
y_pred=tf.add(y7, y8)

learning_rate=0.01
batch_size=1
e=tf.reduce_mean(tf.pow(y_pred-y, 2))
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(e)
init=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    xx1, xx2, yy=get_data()
    step=3
    for i in range(step):
        for j in range(10000//batch_size):
            sess.run(optimizer, feed_dict={x1:xx1[j*batch_size:j*batch_size+batch_size], x2:xx2[j*batch_size:j*batch_size+batch_size], y:yy[j*batch_size:j*batch_size+batch_size]})
    ww1=sess.run(w1)
    ww2=sess.run(w2)
    ww3=sess.run(w3)
    print(ww1, ww2, ww3)