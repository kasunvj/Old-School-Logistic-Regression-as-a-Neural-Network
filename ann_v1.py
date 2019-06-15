#logistic Regression As a Neural Network
#Note:
# in the exel ,value of1A denoted by sheet.cell_value(0,0)
import numpy as np
import xlrd

n_features = 3

	

def sigma(x):
	return 1/(1+np.e ** -x)


def load():
	#print(sheet.cell_value(0,0),sheet.nrows,sheet.ncols) 
	database1 = xlrd.open_workbook("D:/Projects/ANN/dataset1.xlsx")
	sheet = database1.sheet_by_index(0)
	fileds = []
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
	Y = np.transpose(dataset_into_matrix[:,3])
	W = np.zeros((n_features,1))
	b = 0

	Z = np.dot(np.transpose(W),X) + b
	A = sigma(X)

	print(A)

	print(A.shape)

	


def main():
	load()
	



if (__name__ == "__main__"):
	main()
