import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
from sklearn.datasets import load_digits
import logging

class digits_data(object):
	"""This class can load the MNIST digits data from a local directory 
	and if it MNIST data is not available the 8*8 digit images from sklearn is loaded.
	
	public methods: fetch_train
					fetch_test
	public attributes: train: 2D array of training set with each row containing pixel values for a digit
							and 0th column containing the respective labels
						test: 2D array of test set in the same format as training set, 
						but without lables
						test_target: labels of the training set, only exists for scikit learn data set
						loc
	"""

	def __init__(self, loc = '../MNIST/'):
		"""params:
			loc: string, the path to directory with MNIST data, default is '../MNIST/,
				if empty sklearn digits set is loaded
		"""

		self.loc = loc

	def fetch_train(self):
		""" Fetches training data 

		If MNIST 'train.csv' if file is not found, it will load the 8*8 digits dataset from sklearn.
		The sklearn data is divided (80/20) into a training set and test set. 

		Returns:
			train: training data with labels in the 0th column"""

		fpath = self.loc + 'train.csv'

		try:
			with open(fpath) as f:
				print 'Loading MNIST kaggle training set...'
				self.train = pd.read_csv(f, header = 0).values
				return self.train

		except IOError:
			print 'MNIST dataset not found in %s' %self.loc
			digits = load_digits()
			print 'Loading 8*8 digits dataset from scikit...'
			data = np.concatenate((digits.target[:, None], digits.data), axis = 1).astype(np.int16)
			#divide dataset into a training set, test set, and test set targets
			self._train_test_split(data)
			#set location to 'sklearn' indicating data was loaded from scikit_learn
			self.loc = 'sklearn'
			return self.train
	
	def _train_test_split(self, data):
		""" split data (80/20) into a training and test set respectively
			params:
				data: numpy array of data that will be split 
		"""	
		split = np.floor(.8 * len(data))
		self.train = data[0:split]
		self.test = data[split:, 1:]
		self.test_targets = data[split:,0]

	def fetch_test(self):
		""" fetch the test data from directory if using MNIST training data and it was not already fetched
		"""

		if self.loc != 'sklearn':
			fpath = self.loc + 'test.csv'
			with open(fpath) as f:
				print 'Loading MNIST kaggle test set...'
				self.test = pd.read_csv(f)
				return self.test
