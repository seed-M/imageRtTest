

fp=open("test.txt",'r')

lines=fp.readlines()

i=0
for line in lines:
    print(len(line))
    i+=1

print(i)