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

print(X_Train.shape)
print(X_Test.shape)
print(Y_Train.shape)
print(Y_Test.shape)
print("Data Loading Done ...")

class NeuralNetwork:
	def inputLayer(self,input):
		self.input = input
		return input

	def layer(self,prev_mat,number_of_neu, activation ):
		self.prev_mat = prev_mat
		self.number_of_neu = number_of_neu
		self.activation = activation

		n = prev_mat.shape[0]
		m = number_of_neu

		weight = np.zeros((n,m))

		return weight


layer0 = NeuralNetwork().inputLayer(X_Train)
layer1 = NeuralNetwork().layer(layer0,2,"rlu")
layer2 = NeuralNetwork().layer(layer1,3,"rlu")
layer3 = NeuralNetwork().layer(layer2,2,"rlu")

print(layer0.shape)
print(layer1.shape)
print(layer2.shape)
print(layer3.shape)


#layer0 = inputLayer(X_Train)
#layer1 = layer(layer0, number of neurons, activation)
#layer2 = layer(layer1,number of neurons, activation)
#layer3 = layer(layer2, number of neurons, activation)
#layer4 = outputlayer(layer3,Y_Train)

#NeuralNetwork.feedfaward()
#NeuralNetwork.backpropaergation()
#NeuralNetwork.results()





#for i in range (0,iterations):


