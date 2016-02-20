import os
import pandas as pd
import numpy as np
import logging
import naive_bayes as nb
import load_data

def main():
	
	data = load_data.digits('')
	logging.info('training data is loaded')

	#train the classifier
	digitsNB_clf = nb.gaussian_nb(10)
	digitsNB_clf.fit_classifier(data.train_Y, data.train_X)

	logging.info('classifier is trained')
	
	digitsNB_clf.predict(data.values)
	logging.info('done predicting')
	
	pred_error = np.mean(data.test_Y == digitsNB_clf.predictions)
	print pred_error

	digitsNB_clf.predict_to_file('predictions.csv')
	logging.info('done writing predicitons to file')

if __name__ == '__main__':
	logging.basicConfig(filename = 'nb_model.log', filemode = 'w', level = logging.DEBUG)
	main()