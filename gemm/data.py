import numpy as np
n=2048
m=2048
k=1024
a=np.random.random((n,k))
b=np.random.random((k,m))
c=a.dot(b)
file = open("./test","w+",encoding="utf-8")
file.write("{} {} {}\n".format(n,k,m))
a.tofile(file," ")
file.write("\n")
b.tofile(file," ")
file.write("\n")
c.tofile(file," ")
