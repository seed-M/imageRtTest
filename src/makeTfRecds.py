import os
import tensorflow as tf
from PIL import Image

mdir = 'D:/tmp/Images/CatVsDog'
# train.txt
# val.txt
# test.txt
file='train'
clsName={0:'cat',1:'dog'}

tainIdx = open(os.path.join(mdir, file+'.txt'),'r')
writer=[]
map={}
for i,cls in enumerate(clsName):
    map[cls]=i
    writer.append(tf.python_io.TFRecordWriter(os.path.join(mdir, file+'.'+clsName[cls]+'.tfrecords')))
lines = tainIdx.readlines()

i=0
# print("presseed: ",end='')
for line in lines:
    if(len(line)<2):
        print('end')
        continue
    _ = line.split()
    path = _[0]
    cls = _[1]
    img=Image.open(path)
    img=img.resize((227,227))
    img_raw=img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(cls)])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer[map[int(cls)]].write(example.SerializeToString())
    if i%100==0:
        print("have presseed {0} pics".format(i))
    i=i+1

for i,_ in enumerate(clsName):
    writer[i].close()
tainIdx.close()