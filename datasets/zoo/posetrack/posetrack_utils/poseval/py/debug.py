import numpy as np

A = np.random.rand(4,17)

idx = np.where(A[2, :] > 0.3)[0]
#print(A)
#print(idx[0])
#print(A[2,idx])
temp1 = A[:,idx]
x1 = min(temp1[0,:])
x2 = max(temp1[0,:])
y1 = min(temp1[1,:])
y2 = max(temp1[1,:])
print(x1,x2,y1,y2)
