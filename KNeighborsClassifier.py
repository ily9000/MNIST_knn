import math
import pandas as pd
import numpy as np
import logging
import os

class KNN(object):
	""" Classify digit images using the K nearest neighbors classifier 
	To Do: Compute pairwise distance for all points in test set
			and store when determining K from cross validation,
			instead of computing the distance for each K we test.

	public attributes:
	    			data: set during instantiation
				    k_neighbors: set during instantiation
			    	error_k: set by fit_k()
	public methods: 
					fit_k: use cross-validation to determine value for K
					predict: use k nearest neigbhors to return labels for a test set		    	
	"""

	def __init__(self, data, k_neighbors = 3):
		""" Args: 
		data: 2-d numpy array of training data with the labels in the 0th column
		k_neighbors: number of samples nearest to test point used to predict the label
			can either be predetermined or fit using cross validation, defaul is 3
		"""
		
		self.data = data
		self.k_neighbors = k_neighbors

	def _cross_valid(self, k_neighbors):
		""" K-fold cross validation, K must divide number of training samples evenly
		Args:
			k_neighbors: number of neighbors to use for nearest neighbors classifier
		Returns:
			average prediction error across the kfolds
		"""		

		len_subset = len(self.data) / self.kfolds
		#prediction error of the model on each of the validation folds
		error = np.full(self.kfolds, np.nan)

		#iteratively set 1 fold as the validation set and the remaining folds as the training set
		for i in range(self.kfolds):
			valid_set = self.data[(i*len_subset) : (i+1)*len_subset, 1:]
			valid_labels = self.data[(i*len_subset) : (i+1)*len_subset, 0]
			training = np.vstack((self.data[0 : i*len_subset], self.data[(i+1)*len_subset : ]))
			predictions = self.predict(valid_set, training, k_neighbors)
			error[i] = 1 - np.mean(valid_labels == predictions)
		return np.mean(error)

	def fit_k(self, kfolds=10, max_kneighbors=5):
		""" Use cross-validation to learn the value of k for k-nearest neighbors classifier
		Args:
			k_folds: K for K-fold cross validation, default is 10
			max_kneighbors: maximum value to test for k_neighbors, values in range(1,max_kneighbors+1)
				are tested through cross-validation
		"""

		self.kfolds = kfolds
		assert len(self.data) % self.kfolds == 0, \
		"%d-fold validation does not divide number of data points in training set evenly" % self.kfolds
		self.max_kneighbors = max_kneighbors
		self.error_k = np.full(self.max_kneighbors, np.nan)

		for n in range(1, self.max_kneighbors+1):
			self.error_k[n-1] = self._cross_valid(n)
		self.k_neighbors = np.argmax(self.error_k)+1

	def _get_neighbors(self, tr_data, test_val, neighbors):
		""" Computes pairwaise Eulcedian distance between test_val and each point in tr_data 
		Returns:
			indices of the k_neighbors in tr_data closest to test_val """
		
		dist = np.linalg.norm(tr_data-test_val, axis = 1)
		return np.argsort(dist)[0: neighbors]

	def predict(self, test_data, tr_data = None, k_neighbors = None):
		""" Use the training set to find the labels for the test set 
		Args:
			test_data: test set
			tr_data: training set with labels in the 0th column, defaults to instance attribute
			k_neighbors: number of neighbors to use for classifier, defualts to instance attribute 
		Return:
			predictions: np.array with labels for each row in the 2-D test set"""

		if tr_data is None:
			tr_data = self.data
		if k_neighbors is None:
			k_neighbors = self.k_neighbors
		n_predictions = len(test_data)
		predictions = -1 * np.ones(n_predictions, dtype = np.int8)

		#find the k closest inputs in tr_data and return the plurality of their labels
		for j, idx in enumerate(range(n_predictions)):
			neighbors_idx = self._get_neighbors(tr_data[:, 1:], test_data[idx, :], k_neighbors)
			counts = np.bincount(tr_data[neighbors_idx, 0])
			predictions[j] = np.argmax(counts)
		return predictions