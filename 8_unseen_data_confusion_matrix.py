from numpy import loadtxt
from numpy import savetxt
import numpy
from keras.models import load_model
from numpy.random import seed
from sklearn.preprocessing import RobustScaler
import pafy
import cv2
from tensorflow.random import set_seed
import time
from sklearn.metrics import confusion_matrix

# setting the seed
seed(1)
set_seed(1)

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

frame_eeg = loadtxt('E:\\recording-car\\aaryan_time_synced_signals_with_frames.csv', delimiter=',', skiprows=1)

# Create epoch data without labels as required for unseen data

total_epochs = numpy.empty([0, 20])	
frame_iterations = len(frame_eeg) - 5

for frame in frame_eeg[0:frame_iterations, 0]:

	flattened_array = numpy.append(frame_eeg[int(frame)+0, 1:5].reshape(1, 4), frame_eeg[int(frame)+1, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+2, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+3, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+4, 1:5].reshape(1, 4), axis=1)
	 
	total_epochs = numpy.append(total_epochs, flattened_array.reshape(1, 20), axis=0)

savetxt("unseen_total_epochs_20.csv", total_epochs, delimiter=",")

# Data Pre-processing - scale data using robust scaler

input = rScaler.fit_transform(total_epochs[:, 0:20])
input = input.reshape(len(input), 1, 20)
input = input.transpose(0, 2, 1)

# load the model
model = load_model('E:\\recording-car\\model_conv1d.h5')

# get the "predicted class" outcome
y_hat = model.predict(input) 
y_pred = numpy.argmax(y_hat, axis=-1)
print(y_pred.shape)
#y_pred = numpy.reshape(len(y_pred), 1)


epochs = loadtxt('E:\\recording-car\\aaryan_total_epochs_20.csv', delimiter=',')
y_test = epochs[0:498, -1]
matrix = confusion_matrix(y_test, y_pred)
print(matrix)


