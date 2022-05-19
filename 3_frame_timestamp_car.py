# Import necessary libraries
import cv2
import time
import numpy
from numpy import savetxt 
import pafy

# set path in which you want to save images
csv_file = "my_recording.csv"

# Open the link
play = pafy.new('https://www.youtube.com/watch?v=X-OB_lUlF2g&ab_channel=TheHallofAdvertising').streams[-1]
#play = pafy.new('https://www.youtube.com/watch?v=-AffEV6QlyY&ab_channel=LuvJohnnyDepp4ever').streams[-1]
video = cv2.VideoCapture(play.url)

frame_id = 0;
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
	
	data[0] = frame_id
	data[1] = my_timestamp

	data_list = numpy.append(data_list, data.reshape(1, 2), axis=0)

	frame_id = frame_id + 1

	# Display the image
	if img is None:
		break
	else: 
		cv2.imshow('live video', img)

	# wait for user to press any key
	key = cv2.waitKey(100)

# close the camera
video.release()

# close open windows
cv2.destroyAllWindows()

numpy.savetxt(csv_file, data_list, delimiter=",", header="frame_id, timestamp")