#logistic Regression As a Neural Network
#Note:
#This is a basic perceptron 
#perception comes with stimulation. when an eye see the light, eye get stimulated and build the perception called vision
#This gets a stimulus and percive value of the summation 

import numpy as np
import xlrd
import math

n_features = 3
alpha = 0.5

	

def sigma(x):
	return 1/(1+np.e ** -x)


def load():
	#print(sheet.cell_value(0,0),sheet.nrows,sheet.ncols) 
	database1 = xlrd.open_workbook("D:/Projects/Git2/Old-School-Logistic-Regression-as-a-Neural-Network/dataset1.xlsx")
	sheet = database1.sheet_by_index(0)
	fileds = []

	m = sheet.nrows 
	m_traning = math.floor(m*0.75)
	m_testing = math.floor(m*0.25)


	training_set = np.zeros((m_traning,sheet.ncols))
	testing_set = np.zeros((m_testing,sheet.ncols))

	for i in range(0,m_traning):
		for j in range(0,sheet.ncols):
			try:
				training_set[i][j] = sheet.cell_value(i,j)
			except:
				fileds.append(sheet.cell_value(i,j))
	p=0;

	for i in range(m_traning,m):
		p = p + 1
		for j in range(0,sheet.ncols):
			try:
				testing_set[p][j] = sheet.cell_value(i,j)
			except:
				fileds.append(sheet.cell_value(i,j))
	print(training_set)
	print(testing_set)

	training_set = np.delete(training_set,0,0)
	testing_set = np.delete(testing_set,0,0)

	training_dataset_shape = training_set.shape
	testing_data_shape = testing_set.shape

	X_train = np.transpose(np.delete(training_set,n_features,1))
	X_test = np.transpose(np.delete(testing_set,n_features,1))

	Y_train = np.reshape(np.transpose(training_set[:,3]), (1,m_traning -1))
	Y_test = np.reshape(np.transpose(testing_set[:,3]), (1,m_testing -1))


	W = np.zeros((n_features,1))
	b_traning = np.zeros((1,m_traning-1))
	b_testing  = np.zeros((1,m_testing-1))

	for i in range (0,100):
		Z = np.dot(np.transpose(W),X_train) + b_traning
		A = sigma(Z)

		dz = A - Y_train
		dw = np.dot(X_train,np.transpose(dz))/m_traning
		db = np.sum(dz)/m_traning


		W = W - alpha*dw
		b_traning = b_traning - alpha*db

		#calculating the accurasy
		b_testing.fill(b_traning[0][0])
		Z_hat = np.dot(np.transpose(W),X_test) + b_testing
		A_hat = np.round(sigma(Z_hat))




	print("################## Network ######################")
	print("Training set:            ",m_traning)
	print("Testing set:             ",m_testing)
	print("Shapes--------------------------------------------")
	print("dL/dz:                   ",dz.shape)
	print("dw=(1/m)*X*transpose(dz):",dw.shape)
	print("X_train:                 ",X_train.shape)
	print("X_test:                  ",X_test.shape)
	print("Z:                       ",Z.shape)
	print("A:                       ",A.shape)
	print("b_train:                 ",b_traning.shape)
	print("b_test:                  ",b_testing.shape)
	print("Y_train:                 ",Y_train.shape)
	print("--------------------------------------------------")
	print("Wights:                  ",W)
	print("b:                       ",b_traning[0,0])

	print(A)
	print(A_hat)


	


def main():
	load()
	



if (__name__ == "__main__"):
	main()
