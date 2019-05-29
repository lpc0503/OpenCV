import cv2
import numpy as np

lower_color = np.array([150,50,50])
upper_color = np.array([180,255,255])

cap = cv2.VideoCapture(0)

def get_XY():
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# Threshold the HSV image to get only skin colors
	mask = cv2.inRange(hsv, lower_color, upper_color)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	mask = cv2.erode(mask, kernel, iterations = 2)
	mask = cv2.dilate(mask, kernel, iterations = 4)

	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = (0, 0)
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		if radius > 100:
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	return center

if __name__ == "__main__":
	while(1):
		print(get_XY())