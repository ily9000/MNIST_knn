import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
import logging

class digits(object):
	"""This class can load the MNIST digits data from a csv file in a local directory (kaggle format) 
	and if MNIST data is not available the 8*8 digit images from sklearn is loaded.
	
	public methods: 
		fetch_train
		fetch_test: only used if MNIST data is a csv file in local directory
	public attributes: 
		train: 2D array of training set with each row containing pixel values for a digit
				and 0th column containing the respective labels
		test: 2D array of test set in the same format as training set, 
				but without lables
		test_target: labels of the training set, only exists for scikit learn data set
		loc: path directory of the data
	"""

	def __init__(self, loc = '../MNIST/'):
		"""params:
			loc: string, the path to directory with MNIST data, default is '../MNIST/,
				if empty sklearn digits set is loaded
		"""

		self.loc = loc
		self._fetch_train()
		if self.loc != 'sklearn':
			self._fetch_test()

	def _fetch_train(self):
		""" Fetches training data 

		If MNIST 'train.csv' if file is not found, it will load the 8*8 digits dataset 
		from sklearn. The sklearn data is divided (80/20) into a training set and test set. 

		Returns:
			train_X: training features
			train_Y: training labels
		"""
		fpath = self.loc + 'train.csv'

		try:
			with open(fpath) as f:
				print 'Loading MNIST training set form local directory...'
				self.train = pd.read_csv(f, header = 0).values
			self.train_X = self.train[:,0]
			self.train_Y = self.train[:,1:]
			return self.train

		except IOError:
			print 'MNIST dataset not found in directory'
			print 'Loading MNIST dataset from scikit...'
			digits = fetch_mldata('MNIST original')
			#digits = load_digits()
			data = np.concatenate((digits.target[:, None], digits.data), axis = 1)
			#divide dataset into a training set, test set, and test set targets
			np.random.shuffle(data)
			self._train_test_split(data)
			#set location to 'sklearn' indicating data was loaded from scikit_learn
			self.loc = 'sklearn'
			self.train_X = self.train[:,0]
			self.train_Y = self.train[:,1:]
			return self.train_X, self.train_Y
	
	def _train_test_split(self, data):
		""" split data (80/20) into a training and test set respectively """

		split = np.floor(.8 * len(data))
		self.train = data[0:split]
		self.test_X = data[split:, 1:]
		self.test_Y = data[split:,0]

	def _fetch_test(self):
		""" Fetch the test data from directory"""

		fpath = self.loc + 'test.csv'
		with open(fpath) as f:
			print 'Loading MNIST test set...'
			self.test = pd.read_csv(f).values