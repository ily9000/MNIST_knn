from __future__ import division
from scipy.stats import norm
import numpy as np
import pandas as pd
import os

class gaussian_nb():
	"""A Gaussain naive bayes classifier for the MNIST digits data set

	Public Methods:
		fit_classifier: Learn the prior and gaussian parameters for each digit 
			from training set
		predict: Use the fitted classifier to predict the digits of the test samples
		predict_to_file: Write the predictions to a file
	"""

	def __init__(self, N_labels):
		"""Args:
			N_labels: number of unique labels (How many unique digits are in the training set?)
		"""
		self.means = []
		self.idx_0var = []
		self.predictions = []
		self.stdev = []
		self.N_labels = N_labels

	def _update_prior(self, labels):
		"""Set the prior probability of each label by
		counting the number of times the label appears in training data and 
		normalizing it by the total number of training samples.

		Args:
			labels (1D np.array): The labels of the training set.
		"""
		prior_cnts = np.unique(labels, return_counts = True)[1]
		self.prior = prior_cnts/np.sum(prior_cnts)
		
	def fit_classifier(self, data, labels):
		"""Train the classifier on digits data
		Args:
			labels (1D np.array): is an array with the respective labels for each training sample
			data (2D np.array): Each row contains the pixel values of a test image.
		"""
		#number of features (pixels) per image
		self.N_features = data.shape[1]
		assert self.N_labels == len(np.unique(labels)), \
		"Unique labels in training set: %d but this classifer expects: %d" \
		% (len(np.unique(labels)), self.N_labels) 

		self._update_prior(labels)

		#iterate through each digit and get conditional mean and variance for each feature
		for i in range(self.N_labels):
			self.means.append(data[labels == i].mean(axis = 0))
			self.stdev.append(data[labels == i].std(axis = 0))

		self.means = np.vstack(self.means)
		self.stdev = np.vstack(self.stdev)

		#substitute the 0 standard deviation values with the smallest nonzero value in the data
		min_stdev = np.min(self.stdev[self.stdev != 0])
		self.stdev[self.stdev == 0] = min_stdev
					
	def predict(self, data, reset = True):
		"""Predict the label (digit) for each image using maximum posteriori
			and naive bayes conditional independence assumptions. 
		Arguments: 
			data (2D np.array): test set where each row is the pixel values of an image
			reset (bool): Delete the existing predicitons
		Returns:
			integer list where the integers are the predicted digits of each image.
		"""
		assert data.shape[1] == self.N_features, 'Number of features in training set: %d\
		does not match number of features in test set: %d' % (self.N_features, data.shape[0])

		if reset:
			self.predictions = []

		for row in data:
			self.predictions.append(self._argmax_posterior(row))
		return self.predictions

	def predict_to_file(self, f):
		"""Write the predictions attribute to a file in the format "image_ID,digit" 
		with a header "ImageId,Label".
		Args: 
			f (str): The file which will be created and the predictions will be written to.
		"""
		with open(f, 'w') as f:
			f.write('ImageId,Label\n')
			for imageid, value in enumerate(self.predictions):
				f.write('%d,%d\n' % (imageid+1, value))

	def _argmax_posterior(self, sample):
		"""Given 1 test sample, return the digit which maximizes the posteriori
		Args:
			sample (1D np.array): Pixel values of a single image
		Return:
			Digit which maximizes posterior distribution for the test sample
		"""
		posterior = np.sum(norm.logpdf(np.tile(sample, (10,1)),
			loc = self.means, scale = self.stdev), axis = 1) + np.log(self.prior)
		#print posterior
		#print np.argmax(posterior)
		return np.argmax(posterior)
