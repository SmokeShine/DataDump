
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

n_classes=10
img_width=28
img_height=28
initializer = tf.contrib.layers.xavier_initializer()
W={"h1":tf.Variable(initializer([3,3,1,32]),name="W"),
   "h2":tf.Variable(initializer([3,3,32,64])),
   "h3":tf.Variable(initializer([12*12*64,128])),
   "out":tf.Variable(initializer([128,n_classes]))
  }


b={"b1":tf.Variable(tf.zeros([32]),name="W"),
   "b2":tf.Variable(tf.zeros([64])),
   "b3":tf.Variable(tf.zeros([128])),
   "bout":tf.Variable(tf.zeros([n_classes]))}
x=tf.placeholder("float",[None,img_width*img_height])
y=tf.placeholder("float",[None,n_classes])
x1=tf.reshape(x,[30,28,28,1])
l1=tf.nn.conv2d(x1,W["h1"],strides=[1,1,1,1],padding="VALID")
l1_b=tf.nn.bias_add(l1,b["b1"])

l1_act=tf.nn.relu(l1_b)
l2=tf.nn.conv2d(l1_act,W["h2"],strides=[1,1,1,1],padding="VALID")
l2_b=tf.nn.bias_add(l2,b["b2"])
l2_act=tf.nn.relu(l2_b)
l3_max=tf.nn.max_pool(l2_act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")
l4=tf.reshape(l3_max,[30,12*12*64])
l4=tf.matmul(l4,W["h3"])
l4_bias=tf.nn.bias_add(l4,b["b3"])
l4_act=tf.nn.relu(l4_bias)
l5=tf.matmul(l4,W["out"])+b["bout"]
oct=tf.nn.softmax(l5)
print(oct)
print("---------------------------------------------")
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("+++++++++++++++++++++++++++++++++++")
    for epoch in range(100):
        x_raw,y_raw=mnist.train.next_batch(30)
        k=sess.run(optimizer,feed_dict={x:x_raw,y:y_raw})
    pred=oct.eval({x:x_raw})
    print(pred[1:5])
    print(y_raw[1:5])
    print(W["h1"].eval())


