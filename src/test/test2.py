import tensorflow as tf


a=list(range(103))
writer= tf.python_io.TFRecordWriter('test.tfrecords')
for i in a:

    tmp=bytes(str(i+1),encoding='utf8')
    example = tf.train.Example(features=tf.train.Features(feature={
        "plus1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp])),
        "value": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
    }))
    writer.write(example.SerializeToString())
writer.close()