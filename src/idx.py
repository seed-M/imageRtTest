
import os
import random

rootdir='E:/data/DogvsCat'
# imagedir='train/'

ftrain=open(os.path.join(rootdir,'train.idx'),'w')
fval=open(os.path.join(rootdir,'val.idx'),'w')
ftest=open(os.path.join(rootdir,'test.idx'),'w')

for parent,_,filenames in os.walk(rootdir):
    for name in filenames:
        if os.path.splitext(name)[1]!='.jpg':
            continue

        rnd=random.uniform(0,1)
        fp=ftrain
        if rnd<0.3:
            if random.randint(0,1)==0:
                fp=fval
            else:
                fp=ftest
        if name.startswith('cat'):
            fp.write(os.path.join(parent,name)+'\t0'+'\n')
        elif name.startswith('dog'):
            fp.write(os.path.join(parent,name)+'\t1'+'\n')

ftrain.close()
fval.close()
ftest.close()