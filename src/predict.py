# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
import numpy as np
import cv2 as cv
import pickle

#Process the image
def imgproc(filepath):
	img = cv.imread(filepath)
	hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
	
	bg = hsv[0][0]

	hsv_lower = np.zeros(3, dtype=np.uint8)
	hsv_upper = bg
	mask = cv.inRange(hsv, hsv_lower, hsv_upper)
	result = cv.bitwise_and(hsv, hsv, mask=mask)
	
	kernel = np.ones((3, 3), np.uint8)
	dlt = cv.dilate(result, kernel, iterations=5)

	h, s, v = cv.split(dlt)
	thresh = cv.threshold(v, 230, 255, cv.THRESH_BINARY_INV)[1]

	threshT = thresh.T

	cposh = []
	n = threshT.shape[0] - 1

	for i in range(0, n):
		allzero1 = np.any(threshT[i])
		allzero2 = np.any(threshT[i+1])
		if (allzero1 and (not allzero2)) or ((not allzero1) and allzero2):
		    cposh.append(i)


	cposv = []
	n = thresh.shape[0] - 1

	for i in range(0, n):
		allzero1 = np.any(thresh[i])
		allzero2 = np.any(thresh[i+1])
		if (allzero1 and (not allzero2)) or ((not allzero1) and allzero2):
		    cposv.append(i)
	
	up, down = cposv[0]-1, cposv[1]+2


	c1 = thresh[up:down, cposh[0]-1:cposh[1]+2]
	c2 = thresh[up:down, cposh[2]-1:cposh[3]+2]
	c3 = thresh[up:down, cposh[4]-1:cposh[5]+2]

	dim = (20, 20)
	c1 = cv.resize(c1, dim, interpolation=cv.INTER_AREA)
	c2 = cv.resize(c2, dim, interpolation=cv.INTER_AREA)
	c3 = cv.resize(c3, dim, interpolation=cv.INTER_AREA)

	return c1, c2, c3

def decaptcha( filenames ):
	model = pickle.load(open('./model', 'rb'))

	templist = [c for filepath in filenames for c in imgproc(filepath)]
		
	X = np.array(templist)
	nsamples, nx, ny = X.shape
	X = X.reshape((nsamples, nx*ny))
	predictions = model.predict(X)
	predictions = predictions.reshape(-1, 3)
	labels = [','.join(elem) for elem in predictions]
	
	return labels
