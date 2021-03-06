#logistic Regression As a Neural Network
#Note:
# in the exel ,value of1A denoted by sheet.cell_value(0,0)
import numpy as np
import xlrd

n_features = 3
alpha = 0.5

	

def sigma(x):
	return 1/(1+np.e ** -x)


def load():
	#print(sheet.cell_value(0,0),sheet.nrows,sheet.ncols) 
	database1 = xlrd.open_workbook("D:/Projects/Git2/Old-School-Logistic-Regression-as-a-Neural-Network/fix_dataset2.xlsx")
	sheet = database1.sheet_by_index(0)
	fileds = []
	m = sheet.nrows -1  

	dataset_into_matrix = np.zeros((sheet.nrows,sheet.ncols))
	for i in range(0,sheet.nrows):
		for j in range(0,sheet.ncols):
			try:
				dataset_into_matrix[i][j] = sheet.cell_value(i,j)
			except:
				fileds.append(sheet.cell_value(i,j))
	dataset_into_matrix = np.delete(dataset_into_matrix,0,0)
	dataset_shape = dataset_into_matrix.shape

	X = np.transpose(np.delete(dataset_into_matrix,n_features,1))
	Y = np.reshape(np.transpose(dataset_into_matrix[:,3]), (1,m))
	W = np.zeros((n_features,1))
	b = np.zeros((1,m))

	for i in range (0,1000):
		Z = np.dot(np.transpose(W),X) + b
		A = sigma(Z)

		dz = A - Y
		dw = np.dot(X,np.transpose(dz))/m
		db = np.sum(dz)/m

		W = W - alpha*dw
		b = b - alpha*db



	print("Shapes------")
	print(dw)
	print(dz.shape)
	print(dw.shape)
	print(X.shape)
	print(Z.shape)
	print(A.shape)
	print(b.shape)
	print(Y.shape)
	print(W)
	print(b)

	


def main():
	load()
	



if (__name__ == "__main__"):
	main()
