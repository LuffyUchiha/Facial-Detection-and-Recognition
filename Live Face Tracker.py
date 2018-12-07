import os
import numpy as np
import cv2

def draw_rect(img,rect):
	(x,y,w,h)=rect
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



def detect_face(img):
	if img is not None:
		im_cp=img.copy()
		gray=cv2.cvtColor(im_cp,cv2.COLOR_BGR2GRAY)
		face_cascade=cv2.CascadeClassifier('D:\\Python\\haarcascade_frontalface_default.xml')
		faces=(face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5))
		if len(faces)==0:
			return None
		return faces
			
		

def detect_eyes(img):
	if img is not None:
		im_cp=img.copy()
		gray=cv2.cvtColor(im_cp,cv2.COLOR_BGR2GRAY)
		face_cascade=cv2.CascadeClassifier('D:\\Python\\haarcascade_eye.xml')
		faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
		for single in faces:
				draw_rect(im_cp,single)
				(x,y,w,h)=single
				gray=gray[y:y+w,x:x+h]
				eyes=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
				print(single)
				print(eyes)
				if len(eyes)>0:
					eyes[0][0]+=x
					eyes[0][1]+=y
					draw_rect(im_cp,eyes[0])

		cv2.imshow("Faces",im_cp)
		cv2.waitKey(0)

def opti_face(img):
	faces=detect_face(img)
	opt_face=[]
	for i in faces:
		print("Ori")
		print(i)
		print("se")
		draw_rect(img,i)
		cv2.imshow("ff",img)
		cv2.waitKey(0)
		(x,y,w,h)=i
		face_img=img[y:y+w,x:x+h]
		rect=detect_face(face_img)
		temp=rect
		#cv2.imshow("fff",face_img)
		#cv2.waitKey(0)
		xc=0
		yc=0
		while temp is not None:
			print(temp)
			print(rect)
			icp=face_img.copy()
			draw_rect(icp,rect[0])
			(a,b,c,d)=rect[0]
			xc=xc+a
			yc+=b
			face_img=face_img[b:b+c,a:a+d]
			temp=detect_face(face_img)
			if temp is not None:
				rect=temp
		if rect is None:
			np.append(opt_face,i)
			continue;
		rect[0][0]+=xc
		rect[0][1]+=yc
		np.append(opt_face,rect[0])
	return opt_face
#detect_eyes(cv2.imread("test1.png"))

img=cv2.imread("test.jpg")
faces=opti_face(img)
cv2.imshow("fff",img)
cv2.waitKey(0)