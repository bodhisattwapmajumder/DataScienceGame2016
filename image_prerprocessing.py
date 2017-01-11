import pandas
import cv2
import numpy
import random
import numpy as np
import os

def rotateImage(image, angle, foldername=None, filename=None, typ=None):
    num_rows, num_cols  = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (num_cols, num_rows), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
    return result

f = 'G:/train/flat'

src = 'G:/train/flat/'
des = 'G:/train/flat/'


files = []
for filename in os.walk(f):
    files.append(filename)
files = files[0][2]
for filename in files:
	path2 = filename.split(".jpg")[0]
	img = cv2.imread(src + filename)
	img = rotateImage(img,90)
	img2 = rotateImage(img,180)
	cv2.imwrite(des + filename,img)
	cv2.imwrite(des + path2+"_rotate_180.jpg",img2)
	row,col,ch = img.shape
	img2 = cv2.resize(img, (156,156), interpolation = cv2.INTER_AREA)
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
 	equ = cv2.equalizeHist(img)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	mean = 0
	var = .4
	gauss = np.random.normal(mean,var,(row,col,ch))
	noise = gauss.reshape(row,col,ch)
	img2 = img+noise
	newfile = filename.split(".jpg")[0]+str("_noise.jpg")
	cv2.imwrite(newfile,img2)
	
	
	#creating horizontally flipped images with gaussian noise	
	img2 = cv2.flip(img,1)
	y = random.random()
	if y <= .5:
		img2 = img2 + noise
	newfile = filename.split(".jpg")[0]+str("_2.jpg")
	cv2.imwrite(newfile,img2)

	#creating vertically flipped images with gaussian noise
	img3 = cv2.flip(img,0)
	y = random.random()
	if y <= .5:
		img3 = img3 + noise
	newfile = filename.split(".jpg")[0]+str("_3.jpg")
	cv2.imwrite(newfile,img3)
	
	
	l1 = random.sample(range(1,29),4)
	l2 = random.sample(range(1,29),4)  
	#l1 = [1,1,28,28]
	#l2 = [1,28,1,28]
	for i in range(1,5):
		img3 = img2[l1[i-1]:l1[i-1]+128, l2[i-1]:l2[i-1]+128]
		img3 = cv2.blur(img3,(2,2))
		y = random.random()
		if y <= .5:
			#img3 = cv2.flip(img3,1)
		newfile = filename.split(".jpg")[0]+str("_crop_blur_") + str(i) + str(".jpg")
		cv2.imwrite(newfile,img3)
		img4 = cv2.flip(img3,1)
		img4 = rotateImage(img3,90)
		img5 = rotateImage(img3,-90)	
		cv2.imwrite(des + path2 +"_crop" +str(i)+".jpg",img3)
		cv2.imwrite(des + path2 +"_flip" +str(i)+".jpg",img4)
		cv2.imwrite(path2+"_crop_rotate_270_"+str(i)+".jpg",img5)
		cv2.imwrite(path2+"_crop_flip_"+str(i)+".jpg",img4)

	print '******************** ' + str(files.index(filename)) + " done ********************"


