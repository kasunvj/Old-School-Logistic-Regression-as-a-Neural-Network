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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_features = 3
alpha = 0.5
iterations =500

#graphs 
iterations_for_x = []
accuracy_for_y =[]
weight_for_x = []
b_for_y = []
cost_for_z = []

	

def sigma(x):
	return 1/(1+np.e ** -x)


def run():
	# loading data from exel
	database1 = xlrd.open_workbook("D:/Projects/Git2/Old-School-Logistic-Regression-as-a-Neural-Network/fix_dataset2.xlsx")
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

	training_set = np.delete(training_set,0,0) #Removing the top row
	testing_set = np.delete(testing_set,0,0)  #Removing the top row

	training_dataset_shape = training_set.shape
	testing_data_shape = testing_set.shape

	#Input

	X_Train = np.transpose(np.delete(training_set,n_features,1)) # Removing output column which is last column
	X_Test = np.transpose(np.delete(testing_set,n_features,1))

	#Output

	Y_Train = np.reshape(np.transpose(training_set[:,3]), (1,m_traning -1))
	Y_Test = np.reshape(np.transpose(testing_set[:,3]), (1,m_testing -1))

	
	#Initialization 
	
	#Network Design
	number_of_neurons = [2,2] #[layer 1 nerurons, layer2 neurons]
	Input_neurons = 3
	Output_neurons =1





	#Training --------------------------
	#Layer 1 ########################### 

	In_L1_Training = X_train 
	Out_L1 = np.zeros((2,1)) 
	W1 = np.zeros((3,2))
	b1_Training = np.zeros((1,m_traning-1))

	#Layer 2 ########################### 

	In_L2 = Out_L1
	Out_L2 = np.zeros((2,1))
	w2 = np.zeros((2,2))
	b2_Training = np.zeros((1,m_traning-1))

	#Layer 3 ########################### 

	In_L3 = Out_L2
	Out_L3 = 0
	w3 = np.zeros((2,1))
	b3_Training = np.zeros((1,m_traning-1))

	#Testing-----------------------------
	# Layer 1
	In_L1_Testing = X_test
	b1_Testing  = np.zeros((1,m_testing-1))

	#Layer 2
	b2_Testing  = np.zeros((1,m_testing-1))

	#Layer3
	b3_Testing  = np.zeros((1,m_testing-1))


	for i in range (0,iterations):

		#Layer 1
		Z_L1 = np.dot(np.transpose(W1),X_Train) + b1_Traning # ZL_1 = (
		Out_L1[0][0] = sigma(Z_L1[1][1])
		Out_L1[1][0] = sigma(Z_L1[][])

		Z_L2 = np.dot(np.transpose(W2),In_L1_Training) + b1_Traning
		Out_L2 = sigma(Z_L2)
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

		#dJ/dw3 = (A3-Y)A2
		#dJ/dw2 = (A3-A2)A1
		#dJ/dw1 = (A2-A1)X_train

		#dJ/db3 = (A3-Y)
		#dJ/db2 = (A3-A2)
		#dJ/db1 = (A2-A1)




		dz = A - Y_train #(dJ/dz) J is the cost function defined for the entire dataset 
		dw = np.dot(X_train,np.transpose(dz))/m_traning #(dJ/dw) J is the cost function defined for the entire dataset 
		db = np.sum(dz)/m_traning #(dJ/db) J is the cost function defined for the entire dataset 



		w3 = w3 - alpha*dw3
		w2 = w2 - alpha*dw2
		w1 = w1 - alpha*dw1

		b3_traning = b3_traning - alpha*db3
		b2_traning = b2_traning - alpha*db2
		b1_traning = b1_traning - alpha*db1


		#calculating the accurasy
		b_testing.fill(b_traning[0][0])
		Z_hat = np.dot(np.transpose(W),X_test) + b_testing
		A_hat = np.round(sigma(Z_hat))

		point = 0 ;
		for j in range (0,Y_test.shape[1]):
			if (Y_test[0][j] == A_hat[0][j]):
				point = point + 1

		Accuracy = round((point/m_testing)*100,1)
		iterations_for_x.append(i)
		accuracy_for_y.append(Accuracy)
		cost = 1- Accuracy

		weight_for_x.append(W)
		b_for_y.append(b_traning)
		cost_for_z.append(cost)
   




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

	fig = make_subplots(rows = 1, cols = 2)
	fig.add_trace(go.Scatter(x = iterations_for_x , y = accuracy_for_y),row =1,col =1)
	fig.add_trace(go.Surface(x = weight_for_x , y = b_for_y, z = cost_for_z, colorscale='RdBu', showscale=False),row =1,col =2)
	fig.update_layout(height = 600, width = 800, title_text = "Kasun's DeepMind")
	fig.show()




	


def main():
	run()
	



if (__name__ == "__main__"):
	main()
