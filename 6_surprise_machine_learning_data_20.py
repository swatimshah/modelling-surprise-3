from numpy import loadtxt
from numpy import savetxt
import numpy

frame_eeg = loadtxt('E:\\recording-car\\urvi_time_synced_signals_with_frames.csv', delimiter=',', skiprows=1)

surprise_onset = numpy.empty([16])	

surprise_onset[0] = 286
surprise_onset[1] = 287
surprise_onset[2] = 288
surprise_onset[3] = 289
surprise_onset[4] = 290
surprise_onset[5] = 291
surprise_onset[6] = 292
surprise_onset[7] = 293
surprise_onset[8] = 353
surprise_onset[9] = 354
surprise_onset[10] = 355
surprise_onset[11] = 356
surprise_onset[12] = 357
surprise_onset[13] = 358
surprise_onset[14] = 359
surprise_onset[15] = 360


total_epochs = numpy.empty([0, 21])	
frame_iterations = len(frame_eeg) - 5

for frame in frame_eeg[0:frame_iterations, 0]:

	flattened_array = numpy.append(frame_eeg[int(frame)+0, 1:5].reshape(1, 4), frame_eeg[int(frame)+1, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+2, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+3, 1:5].reshape(1, 4), axis=1)
	flattened_array = numpy.append(flattened_array, frame_eeg[int(frame)+4, 1:5].reshape(1, 4), axis=1)
	 
	for i in range (0, 15):
		if (surprise_onset[i] == int(frame)+0 or surprise_onset[i] == int(frame)+1 or surprise_onset[i] == int(frame)+2 or surprise_onset[i] == int(frame)+3 or surprise_onset[i] == int(frame)+4):
			label = numpy.array([1])
			break
		else:
			label = numpy.array([0])	

	flattened_array = numpy.append(flattened_array, label.reshape(1, 1), axis=1)	
	total_epochs = numpy.append(total_epochs, flattened_array.reshape(1, 21), axis=0)

savetxt("urvi_total_epochs_20.csv", total_epochs, delimiter=",")
