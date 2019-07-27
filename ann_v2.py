#logistic Regression As a Neural Network
#Note:
#This is a basic perceptron 
#perception comes with stimulation. when an eye see the light, eye get stimulated and build the perception called vision
#This gets a stimulus and percive value of the summation 
#What does the logictic regression mean 
#Logistic regression is a statistical model that in its basic form uses a
# logistic function to model a binary dependent variable, although many more
# complex extensions exist. In regression analysis, logistic regression (or 
# logit regression) is estimating the parameters of a logistic model (a form 
# of binary regression). Mathematically, a binary logistic model has a dependent
# variable with two possible values, such as pass/fail which is represented by
# an indicator variable, where the two values are labeled "0" and "1". In the logistic
# model, the log-odds (the logarithm of the odds) for the value labeled "1" 
# is a linear combination of one or more independent variables ("predictors"); 
# the independent variables can each be a binary variable (two classes, coded 
# by an indicator variable) or a continuous variable (any real value). 
# The corresponding probability of the value labeled "1" can vary between
# 0 (certainly the value "0") and 1 (certainly the value "1"), hence the labeling;
# the function that converts log-odds to probability is the logistic function,
# hence the name. -wiki

import numpy as np
import xlrd
import math

n_features = 3
alpha = 0.5
iterations =1000

	

def sigma(x):
	return 1/(1+np.e ** -x)


def load():
	#print(sheet.cell_value(0,0),sheet.nrows,sheet.ncols) 
	database1 = xlrd.open_workbook("D:/Projects/Git2/Old-School-Logistic-Regression-as-a-Neural-Network/dataset3.xlsx")
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

	for i in range (0,iterations):
		Z = np.dot(np.transpose(W),X_train) + b_traning
		A = sigma(Z)

		#math between these lines 
		
		#lost fucntion, J(A,Y_tain) = -(Y_train*log(A) + (1-Y_train)log(1-A)) 
		
		#why this is our loss fuction for logistic regression, 
		# we want loss, J to be smallest as possible
		# when output(A) from NN is very large and the desired output(Y_train) is 1  
		# J = -log(A) ; J will be very small, closer to zero
		# so we can deside there is no loss between the output of NN and desided value
		# when output(A) from NN is very low and the desired output(Y_train) is 0
		# J =-log(1- A) ; J will be very small, coser to zero
		# so we can deside there is no loss between the output of NN and desided value   
		
		# We want to find values for W , B which minimize J

		# z = WX +b      note: Capital leters denote matrix format
		# A = sigma(z)

		# to find W and B which minimize J we need to find, dJ/dw and dJ/db
		# for that we need to find dJ/dz because dJ/dw = dJ/dz*dz/dw
		# to find dJ/dz we need to find dJ/dA because dJ/dz = dJ/dA*dA/dz
		#dJ/dA = -y/a + (1-y)/(1-a)
		
		#dJ/dz = A-Y
		#dJ/dW = (A-Y)X
		#dJ/db = (A-Y)



		dz = A - Y_train #(dJ/dz) J is the cost function defined for the entire dataset 
		dw = np.dot(X_train,np.transpose(dz))/m_traning #(dJ/dw) J is the cost function defined for the entire dataset 
		db = np.sum(dz)/m_traning #(dJ/db) J is the cost function defined for the entire dataset 


		W = W - alpha*dw
		b_traning = b_traning - alpha*db

		#calculating the accurasy
		b_testing.fill(b_traning[0][0])
		Z_hat = np.dot(np.transpose(W),X_test) + b_testing
		A_hat = np.round(sigma(Z_hat))

		point = 0 ;
		for i in range (0,Y_test.shape[1]):
			if (Y_test[0][i] == A_hat[0][i]):
				point = point + 1

		Accuracy = round((point/m_testing)*100,1)
		print("Accuracy:                ",Accuracy,"%")




	print("################## Network ######################")
	print("Training set:            ",m_traning)
	print("Testing set:             ",m_testing)
	print("Shapes--------------------------------------------")
	print("dz=dJ/dz:                ",dz.shape)
	print("dw=(1/m)*X*transpose(dz):",dw.shape)
	print("X_train:                 ",X_train.shape)
	print("X_test:                  ",X_test.shape)
	print("Y_train:                 ",Y_train.shape)
	print("Y_test:                  ",Y_test.shape)
	print("Z:                       ",Z.shape)
	print("A:                       ",A.shape)
	print("Z_hat:                   ",Z_hat.shape)
	print("A_hat:                   ",A_hat.shape)
	print("b_train:                 ",b_traning.shape)
	print("b_test:                  ",b_testing.shape)
	print("--------------------------------------------------")
	print("Wights:                  ",W)
	print("b:                       ",b_traning[0,0])
	print("number of iterations:    ",iterations)
	print("Accuracy:                ",Accuracy,"%")
	print("Just look the similarity of Desired(top) one and predicted one(bottom)")
	print(Y_test)
	print(A_hat)



	


def main():
	load()
	



if (__name__ == "__main__"):
	main()
