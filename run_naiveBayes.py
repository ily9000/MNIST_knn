import os
import pandas as pd
import numpy as np
import logging
import naive_bayes as nb
import load_data

def plot_params(param, fname):
    fig, axes = plt.subplots(1,10)
    for i, ax in enumerate(axes):
        ax.tick_params(labelleft='off',labelbottom='off')
        im = ax.imshow(param[i].reshape(28,28))

    cbar_ax = fig.add_axes([0.33, 0.4, 0.3, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
    plt.rcParams["figure.figsize"] = [25,15]
    plt.savefig(fname, bbox_inches='tight')

def main():
	
	data = load_data.digits('')
	logging.info('training data is loaded')

	#train the classifier
	digitsNB_clf = nb.gaussian_nb(10)
	digitsNB_clf.fit_classifier(data.train_Y, data.train_X)

	logging.info('classifier is trained')
	
	digitsNB_clf.predict(data.test_X)
	logging.info('done predicting')
	
	pred_error = np.mean(data.test_Y == digitsNB_clf.predictions)
	print pred_error

	plot_params(digitsNB_clf.stdev, 'images/NB_stdev.png')
	plot_params(digitsNB_clf.means, 'images/NB_means.png')
	#digitsNB_clf.predict_to_file('predictions.csv')
	#logging.info('done writing predicitons to file')

if __name__ == '__main__':
	logging.basicConfig(filename = 'nb_model.log', filemode = 'w', level = logging.DEBUG)
	main()