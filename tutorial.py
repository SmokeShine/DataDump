
import numpy as np
import tensorflow as tf
def sigmoid(x):
   return 1/(1+np.exp(-x)) 
n_hidden=2
n_input=3
n_output=1

tf.reset_default_graph()
W={"h1":tf.Variable(tf.ones([n_input,n_hidden]),name="W"),
   "out":tf.Variable(tf.ones([n_hidden,n_output]))
  }


b={"b1":tf.Variable(tf.ones([n_hidden])),
   "bout":tf.Variable(tf.ones([n_output]))
  }
x=tf.placeholder("float",([None,n_input]))
y=tf.placeholder("float",([None,n_output]))
l1=tf.add(tf.matmul(x,W["h1"]),b["b1"])

l1_act=tf.sigmoid(l1)
out=tf.add(tf.matmul(l1_act,W["out"]),b["bout"])
out_act=tf.sigmoid(out)

cost=tf.reduce_mean(tf.abs(tf.subtract(out_act,y)))
train_step=tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(cost)
x_raw=np.array([[1,2,3]])
y_raw=np.array([[3]])
y_raw=np.reshape(y_raw,(1,1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred=out_act.eval({x:x_raw,y:y_raw})
    print(pred)
    for epoch in range(100):
        k=sess.run(train_step,feed_dict={x:x_raw,y:y_raw})

print(tf.trainable_variables())
