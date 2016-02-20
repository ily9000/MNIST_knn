<center> #### Implementation of Gaussian Naive Bayes and KNN for digit recoginition </center>

I coded from scratch the Gaussian Naive Bayes and KNN algorithm to classify digits in the MNIST data set.

By assuming conditional independce of the pixels given the digit, the Naive Bayes classifier is able to learn the Gaussian distribution parameters (mean, standard deviation) for each pixel of each digit. A plot of the parameters that were learned is below.

Here are the means of the pixels for each digit:
![](images/NB_means.png?raw=true)


Here are the standard deviations of the pixels for each digit:
![](images/NB_stdev.png?raw=true)

The MNIST data set consists of 72,000. We split 57,600 training images and 14,400 test images. The algorithm is able to learn the parameters and make the predictions in less than a minute.