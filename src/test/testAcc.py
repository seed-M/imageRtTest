import tensorflow as tf



if __name__ == '__main__':

    a=tf.constant([0,0,0,0,0,0,0,0,0,0])
    lb=a%2
    s=[]
    for i in range(9):
        s.append([2.,1.])
    s.append([4,5])

    score=tf.constant(s)
    correct = tf.nn.in_top_k(score, lb, 1)

    acc_op=tf.reduce_mean(tf.cast(correct, tf.int32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        acc=sess.run(acc_op)
        print(acc)