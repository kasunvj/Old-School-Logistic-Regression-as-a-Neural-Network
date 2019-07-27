import numpy as np
for i in range(0,10):
	print(i)

A = np.zeros((1,10)) #10 elements
B = np.zeros((1,10))

for i in range (0,10):
	A[0][i] = i;

B.fill(234)

print(A)#[[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]]
print(A.shape)# (1,10)
print(A.shape[1])

print(A[0][2])#2.0
print(B)

