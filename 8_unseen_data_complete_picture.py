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

frame_eeg = loadtxt('E:\\recording-car\\ritu_time_synced_signals_with_frames.csv', delimiter=',', skiprows=1)

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


epochs = loadtxt('E:\\recording-car\\ritu_total_epochs_20.csv', delimiter=',')
y_test = epochs[0:337, -1]
matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# Open the link
play = pafy.new('https://www.youtube.com/watch?v=X-OB_lUlF2g&ab_channel=TheHallofAdvertising').streams[-1]
video = cv2.VideoCapture(play.url)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # *'mp4v'
videoWriter = cv2.VideoWriter('surprise_detection_subject_ritu.mp4', fourcc, 10, (int(video.get(3))+20,int(video.get(4))+20))

frame_id = 0
surprise_frames = 0
data_list = numpy.empty([0, 2])
data = numpy.empty([2])

while True:
	# Read video by read() function and it
	# will extract and return the frame
	ret, img = video.read()

	# Put current DateTime on each frame
	font = cv2.FONT_HERSHEY_PLAIN
	my_timestamp = round(time.time(), 3)
	cv2.putText(img, str(frame_id), (20, 40),
				font, 2, (255, 255, 255), 2, cv2.LINE_AA)



	row, col = img.shape[:2]
	bottom = img[row-2:row, 0:col]
	mean = cv2.mean(bottom)[0]

	bordersize = 10
	borderred = cv2.copyMakeBorder(
		img,
		top=bordersize,
		bottom=bordersize,
		left=bordersize,
		right=bordersize,
		borderType=cv2.BORDER_CONSTANT,
		value=[0, 0, 255]
	)

	bordersize = 10
	borderblue = cv2.copyMakeBorder(
		img,
		top=bordersize,
		bottom=bordersize,
		left=bordersize,
		right=bordersize,
		borderType=cv2.BORDER_CONSTANT,
		value=[255, 0, 0]
	)


	data[0] = frame_id
	data[1] = my_timestamp

	data_list = numpy.append(data_list, data.reshape(1, 2), axis=0)

	frame_id = frame_id + 1
	if(frame_id > 336):
		break

	# Display the image
	if img is None:
		break
	else:
		if (y_pred[frame_id] == 1): 	
			surprise_frames = surprise_frames + 1	
			cv2.imshow('live video', borderred)
			videoWriter.write(borderred)
		else:
			cv2.imshow('live video', borderblue)	
			videoWriter.write(borderblue)

	# wait for user to press any key
	key = cv2.waitKey(100)

# close the camera
video.release()
videoWriter.release()

# close open windows
cv2.destroyAllWindows()

print(surprise_frames)