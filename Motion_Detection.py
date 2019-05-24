# --------------------------
# Created by Subhashis Suara
#---------------------------

import cv2
import numpy as np

# Video Capture 
capture = cv2.VideoCapture(0) # Webcam
#capture = cv2.VideoCapture(" ") # Video file location
 
fgbg = cv2.createBackgroundSubtractorMOG2(200, 10, True) # History, Threshold, DetectShadows
FrameCount = 0 # Keeps track of what frames

while True:
	# Return Value and the current frame
	retVal, frame = capture.read()

	if not retVal: # Check if a current frame actually exists
		break

	FrameCount += 1
	
	ResizedFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Resize the frame

	mask = fgbg.apply(ResizedFrame) # Get the foreground mask

	count = np.count_nonzero(mask) # Count all the non zero pixels within the mask

	print('Frame: {}, Pixel Count: {}'.format(FrameCount, count))

	if (FrameCount > 1 and count > 1000): # Number of pixels to be considered as movement
		print('Motion detected!')
		cv2.putText(ResizedFrame, 'Motion detected!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	cv2.imshow('Frame', ResizedFrame)
	cv2.imshow('Mask', mask)


	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()