'''Use cross validation to determine K and 
then use K Nearest Neighbor classify array of pixels from digit images

Author: Ilyas Patanam
'''
import numpy as np
from load_data import digits_data
from KNeighborsClassifier import KNN

def mnist_digit_recognition():
	''' predict digits for the MNIST data set'''

	data = digits_data()
	data.fetch_train()
	data.fetch_test()

	mnist_knn = KNN(data.train)
	mnist_knn.fit_k()

	predictions = mnist_knn.predict(data.test)
	return mnist_knn, predictions

def scikit_digit_recognition():
	''' predict digits for scikit learn digits dataset '''

	data = digits_data('')
	data.fetch_train()
	
	sklearn_knn = KNN(data.train)
	#fit k using cross-validation
	sklearn_knn.fit_k(kfolds = 3)

	predictions = sklearn_knn.predict(data.test)
	error = 1 - np.mean(predictions == data.test_targets)

if __name__ == '__main__':
	scikit_digit_recognition()