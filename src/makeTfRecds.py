import os
import tensorflow as tf
from PIL import Image

mdir = '/home/wm/tmp/data/dogvscat'
# train.txt
# val.txt
# test.txt
file='train'
clsName={0:'cat',1:'dog'}

map={}
for i,cls in enumerate(clsName):
    map[cls]=i

tainIdx = open(os.path.join(mdir, file+'.idx'),'r')
lines = tainIdx.readlines()


writer=[None]*len(clsName)
iterNum=[0]*len(clsName)
fileNum=[0]*len(clsName)

# j=0
# l=0
# lines = tainIdx.readlines()
# writer=None
# print("presseed: ",end='')

i=0
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

    if iterNum[map[int(cls)]]%1000==0:
        if (writer[map[int(cls)]] != None):
            writer[map[int(cls)]].close()
        writer[map[int(cls)]]=tf.python_io.TFRecordWriter(os.path.join(mdir, file+'.'+clsName[int(cls)]+str(fileNum[map[int(cls)]])+'.tfrecords'))
        fileNum[map[int(cls)]]+=1

    writer[map[int(cls)]].write(example.SerializeToString())
    iterNum[map[int(cls)]]+=1
    if i%100==0:
        print("have presseed {0} pics".format(i))
    i=i+1

for w in writer:
    w.close()
tainIdx.close()