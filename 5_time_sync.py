from numpy import savetxt
from numpy import loadtxt
import numpy
import matplotlib.pyplot as plt

csv_file = "urvi_time_synced_signals_with_frames.csv"

frame_ts = loadtxt('e:\\recording-car\\my_recording.csv', delimiter=',', skiprows=1)
print(frame_ts[:, 0].shape)

channels_ts = loadtxt('e:\\recording-car\\urvi_gcar_signal_50Hz_outliers_zero_mean.csv', delimiter=',', skiprows=1)
print(channels_ts.shape)

frame_eeg = numpy.empty([0, 5])	

for frame in frame_ts[:, 0]:

	frame_no = int(frame)
	print(frame_no)	

	frame_timestamp = frame_ts[frame_no, 1]	
	i = 0

	while (i < len(channels_ts)):

		eeg_ts = channels_ts[i, 4]
			
		if(eeg_ts > frame_timestamp):
			frame_eeg = numpy.append(frame_eeg, numpy.array([frame_no, channels_ts[i, 0], channels_ts[i, 1], channels_ts[i, 2], channels_ts[i, 3]]).reshape(1, 5), axis=0)
			print(frame_eeg.shape)			
			break
		else:
			i = i + 1
	

savetxt(csv_file, frame_eeg, delimiter=",", header="frame, TP9, AF7, AF8, TP10")