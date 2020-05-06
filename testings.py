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
print(A.shape[1]) #10

print(A[0][2])#2.0
print(B) #[[234. 234. 234. 234. 234. 234. 234. 234. 234. 234.]]

Out_L2 = np.zeros((2,1))
Out_L2[0][0] =123
Out_L2[1][0] =321

print(Out_L2)#[[123.]
             # [321.]]

P = [1,2,3,45,6] 
print(len(P)) # 5
print(P[0]) #1

# Classes ---------------------------------------
class snake :
	name = "pyhon"
	length = "24"
	n= 10
	m= 20
	matrix1 = np.zeros((n,m))

	def change_names(self, new_name,new_length):
		self.name = new_name
		self.length = new_length
	def change_matrix_dim(self,new_n,new_m):
		self.n = new_n
		self.m = new_m


snake_1 = snake() # instantiate


print(snake_1.name)#pyhon
print(snake_1.length)#24
print(snake_1.matrix1.shape)#(10, 20)

snake_1.change_names("Anaconda","56")
print(snake_1.name)#Anaconda
print(snake_1.length)#56

snake_1.change_matrix_dim(1,2)
print(snake_1.matrix1.shape)#(10, 20)
#--------------------------------------------------
class girl :
	def __init__(self,name,age):
		self.name = name
		self.age = age 
	def change_name(self, new_name, new_age):
		self.name = new_name
		self.age = new_age

girly_one = girl("Chalani",28)
girly_two = girl("Madu",22)

print(girly_one.name)#Chalani

#---------------------------------------------------
class MatrixClass():
	def __int__(self):
		self.n = 0
		self.m = 0
	def matrix(self,p,q):
		self.n = p 
		self.m = q
		return np.zeros((p,q))

print(MatrixClass().matrix(2,3)) #[[0. 0. 0.]
                                  #[0. 0. 0.]]

#----------------------------------------------------
xy = np.zeros((2,3))
xy.fill(234)

class MatrixShow():
	def show(self,Mat):
		return Mat\

print(MatrixShow().show(xy))# [[234. 234. 234.]
                              #[234. 234. 234.]]
#----------------------------------------------------






