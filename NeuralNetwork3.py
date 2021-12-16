#####Neural Network Implementation of MNIST Dataset######
######Author: Omey Mohan Manyar##########################
######Email: manyar@usc.edu ############################


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from ANN import *
import time
import sys

import pandas as pd

t0 = time.time()
image_size = 28     #width and length of the mnist dataset
no_of_labels = 10    #0,1,2,.....,9 labels in the dataset

image_pixels = image_size * image_size


training_filename = sys.argv[1]
train_label_filename = sys.argv[2]

testing_filename = sys.argv[3]
#test_label_filename = "test_label.csv"


#training_filename = "train_image.csv"
#train_label_filename = "train_label.csv"

#testing_filename = "test_image.csv"
#test_label_filename = "test_label.csv"

#training_data = np.asarray(np.loadtxt("train_image.csv", delimiter=","))      #Training images with at most 60000 images and 784 pixels
training_data = np.asfarray(pd.read_csv(training_filename, header = None))      #Training images with at most 60000 images and 784 pixels


#training_labels = np.asarray(np.loadtxt("train_label.csv", delimiter=","))      #Training labels for the corresponding training images
training_labels =  np.asfarray(pd.read_csv(train_label_filename, header = None))      #Training labels for the corresponding training images

#test_data = np.asarray(np.loadtxt("test_image.csv", delimiter=","))           #Testing images with at most 60000 images and 784 pixels
test_data =  np.asfarray(pd.read_csv(testing_filename, header = None))           #Testing images with at most 60000 images and 784 pixels


#test_labels = np.asarray(np.loadtxt("test_label.csv", delimiter=","))      #Testing labels for the corresponding testing images
#test_labels =  np.asfarray(pd.read_csv(test_label_filename, header = None))      #Testing labels for the corresponding testing images



#Checking input
#print(training_data.shape)
#print(training_labels.shape)
#print(test_data.shape)

#print(training_labels[0])

#Normalizing data to avoid 0 inputs
mul_factor = 0.99/255

training_images = training_data[:, 0:] * mul_factor + 0.01
test_images = test_data[:,0:] * mul_factor + 0.01


#converting the labels into a vectorized form
label_range = np.arange(no_of_labels)

training_labels_mat = (label_range == training_labels).astype(np.float)
#test_labels_mat = (label_range == test_labels).astype(np.float)

#normalizing and removing zeros and ones
training_labels_mat[(training_labels_mat == 0)] = 0.01
training_labels_mat[(training_labels_mat == 1)] = 0.99

#test_labels_mat[(test_labels_mat == 0)] = 0.01
#test_labels_mat[(test_labels_mat == 1)] = 0.99



#Checking Input
#for i in range(10):
#    img = training_images[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()

#for i in range(10):
#    img = test_images[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()


print("Defining the Network Architecture")
Network = Neural_Network(network_structure= [image_pixels, 250,125,10], learning_rate = 0.01, bias = 0.1)
print("Training...........")
Network.train(training_images, training_labels_mat, epochs = 12)
#Network.train_mini_batch(training_images, training_labels_mat, batch_size= 10,epochs = 3)
print("Training Complete")


corrects,wrongs = Network.evaluate(training_images, training_labels)

print("Training Accuracy = ", corrects*100/(corrects+wrongs))

#corrects,wrongs = Network.evaluate(test_images, test_labels)

#print("Testing Accuracy = ", corrects*100/(corrects+wrongs))

test_predictions = Network.get_predictions(test_images)

np.savetxt('test_predictions.csv',test_predictions,delimiter=',', fmt='%d')
t1 = time.time()

print("Time taken for execution = ", t1-t0)