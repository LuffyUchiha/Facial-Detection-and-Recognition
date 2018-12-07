import os
import numpy as np
import cv2

subjects=[""]

con=70
recog=cv2.face.LBPHFaceRecognizer_create()


def distance(a,b,x,y):
	t=(a-x)*(a-x)+(b-y)*(b-y)
	return t

def is_inside(rect1,rect2):
	(a,b,c,d)=rect1
	(x,y,z,w)=rect2
	if a<=x and b<=y and a+c>=x+z and b+d>=y+w:
		return True
	else:
		return False

def is_outside(rect1,rect2):
	(a,b,c,d)=rect1
	(x,y,z,w)=rect2
	if a>=x and b>=y and a+c<=x+z and b+d<=y+w:
		return True
	else:
		return False

def in_radius(rect1,rect2):
	#print(rect1)
	#print(rect2)
	(a,b,c,d)=rect1
	(x,y,w,h)=rect2
	d1=distance(a,b,x,y)
	d2=distance(a+c,b+d,x+w,y+h)
	d3=distance(a+c,b,x+w,y)
	d4=distance(a,b+y,x,y+w)
	if d1<=100 or d2<=100 or d3<=100 or d4<=100:
		return True
	else:
		return False


def is_not_similar(face,rect):
	for i in face:
		if in_radius(rect,i) or is_inside(rect,i) or is_outside(rect,i):
			return False
	return True

def appnd(face,face1):
	for i in face1:
		if is_not_similar(face,i):
			face=np.vstack((face,i))
	return face

def draw_rect(img,rect):
	(x,y,w,h)=rect
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


def draw_text(img,text,x,y):
	cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)

def face_vision(img,l):
	if img is not None:
		im_cp=img.copy()
		gray=cv2.cvtColor(im_cp,cv2.COLOR_BGR2GRAY)
		face_cascade=cv2.CascadeClassifier('D:\\Python\\haarcascade_frontalface_default.xml')
		if l==1:
			face_cascade1=cv2.CascadeClassifier('D:\\Python\\haarcascade_frontalface.xml')
			face_cascade2=cv2.CascadeClassifier('D:\\Python\\haarcascade_frontalface_alt.xml')
			face_cascade3=cv2.CascadeClassifier('D:\\Python\\haarcascade_frontalface_alt2.xml')
			face_cascade4=cv2.CascadeClassifier('D:\\Python\\haarcascade_profileface.xml')
		faces=(face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5))
		if l==1:
			if len(faces)==0:
				faces=[(0,0,0,0)]
			faces=appnd(faces,(face_cascade1.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)))
			faces=appnd(faces,(face_cascade2.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)))
			faces=appnd(faces,(face_cascade3.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)))
			faces=appnd(faces,(face_cascade4.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)))
		if len(faces)==0:
			return None
		else:
			return faces

def detect_face(img):
	im_cp=img.copy()
	gray=cv2.cvtColor(im_cp,cv2.COLOR_BGR2GRAY)
	faces=face_vision(im_cp,0)
	if np.any(faces==(0,0,0,0)):
		faces=np.delete(faces,0,0)
	if faces is None:
		return None
	elif (len(faces)==1) :
		return faces
	else:
		return faces

def face_detection(img):
	faces=face_vision(img,0)
	if np.all(faces==[(0,0,0,0)]):
		return None
	else:
		return faces



def data_prepare(folder_path):
	try:
		dirs=os.listdir(folder_path)
	except FileNotFoundError:
		fol_dir=input("The directory doesn't exist. PLease try again\n")
		faces,labels=data_prepare(fol_dir)
		return faces,labels
	face_list=[]
	labels=[]
	print("Please wait. Training")
	for dir_name in dirs:
		path=folder_path+"/"+dir_name
		images=os.listdir(path)
		for image in images:
			print(image)
			if image.startswith("."):
				continue;
			im_path=path+"/"+image
			test_img=cv2.imread(im_path)
			print(".")
			faces=detect_face(test_img)
			if faces is None:
				continue;
			for i in faces:
				img=test_img.copy()
				gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				(x,y,w,h)=i
				face=gray[y:y+w,x:x+h]
				face_list.append(face)
				draw_rect(img,i)
				cv2.namedWindow("Who is this?",cv2.WINDOW_AUTOSIZE)
				cv2.resizeWindow("Who is this?", 1280,720)
				cv2.imshow("Who is this?",img)
				k=cv2.waitKey(0)
				cv2.destroyAllWindows()
				name=input("Enter his name\n")
				if name not in subjects:
					label=len(subjects)
					labels.append(label)
					subjects.append(name)
				else:
					label=subjects.index(name)
					labels.append(label)

				cv2.destroyAllWindows()
				cv2.waitKey(1)
				cv2.destroyAllWindows()
	print(face_list,labels)
	return face_list,labels


cv2.imshow("Hello",cv2.imread("878324.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Hello")
pur=input("Would you like to load an existing classifier(l) or build a new classifier(b)\n")
if pur=='l':
	classifier=input("Enter the name of the classifier. No need for extension\n")
	f=open(classifier+".txt")
	f1=f.readlines()
	for x in f1:
		subjects=x.split(',')
	print(subjects)
	recog.read(classifier+".xml") #recog.load() if opencv 2._

else:
	train_directory=input("Enter the directory to train\n")
	faces,labels=data_prepare(train_directory)
	print("Faces:",len(faces))
	print("Labels:",len(labels))
	recog.train(faces,np.array(labels))


def retrain(test_img,rect):
	nf=[]
	nl=[]
	if rect is None:
		return
	img=test_img.copy()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	(x,y,w,h)=rect[0]
	face=gray[y:y+w,x:x+h]
	nf.append(face)
	draw_rect(img,rect[0])
	cv2.namedWindow("Who is this?",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Who is this?", 1280,720)
	cv2.imshow("Who is this?",img)
	cv2.waitKey(0) & 0xFF #only for 64-bit systems
	cv2.destroyAllWindows()
	name=input("Enter his name\n")
	if name not in subjects:
		label=len(subjects)
		nl.append(label)
		subjects.append(name)
		recog.update(nf,np.array(nl))
	elif name == "":
		return
	else:
		label=subjects.index(name)
		nl.append(label)
		recog.update(nf,np.array(nl))

def predict(test_img):
	img=test_img.copy()
	rect=face_detection(img)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	if rect is None:
		print("Face not recognized. Please try uploading another picture\n")
	elif(len(rect)==1):
		(x,y,w,h)=rect[0]
		face=gray[y:y+w,x:x+h]
		label,confidence=recog.predict(face)
		if confidence>con:
			retrain(test_img,rect)
		else:
			label_text=subjects[label]
			draw_rect(img,rect[0])
			draw_text(img,label_text,x,y-5)
	else:
		current_max=[500]*len(subjects)
		current_locs=[(0,0,0,0)]*len(subjects)
		for single in rect:
			(x,y,w,h)=single
			label,confidence=recog.predict(gray[y:y+w,x:x+h])
			if(confidence>con):
				retrain(test_img,single)
			elif(current_max[label]>confidence):
				current_max[label]=confidence
				current_locs[label]=single
		i=0
		for vals in current_locs:
			if i==0:
				i+=1
				continue;
			draw_rect(img,vals)
			draw_text(img,subjects[i],vals[0],vals[1]-5)
			i+=1		
	cv2.namedWindow(subjects[1],cv2.WINDOW_NORMAL)
	cv2.resizeWindow(subjects[1], 1280,720)
	cv2.imshow(subjects[1],img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#test_img1=cv2.imread("test1.png")
#test_img2=cv2.imread("test2.jpg")
#predict(test_img1)
#res2=predict(test_img2)
#rint("Hello "+res1)



while True:
	result=input("Would you like to train the model more(t) or recognizing face in another picture(r). Press any other key to quit\n")
	if result=='t':
		x=True
		while x:
			filepath=input("Enter the path directory of the picture. Press q if same directory\n")
			if filepath=='q':
				break;
			try:
				os.chdir(filepath)
				x=False
			except FileNotFoundError:
				print("File not found. Please try again\n")
		images=os.listdir(filepath)
		for image in images:
			if image.startswith("."):
				continue;
			im_path=filepath+"/"+image
			train_image=cv2.imread(im_path)
			if train_image is None:
				continue;
			#train_image=cv2.resize(train_image,(0,0),fx=1000/train_image.shape[1],fy=1000/train_image.shape[0],interpolation=cv2.INTER_AREA)
			img=train_image.copy()
			print("Please Wait....")
			rect=face_detection(img)
			if rect is not None:
				for face in rect:
					retrain(img,face)
			#gray=cv2.cvtColor(train_image,cv2.COLOR_BGR2GRAY)
			#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
			#cv2.waitKey(0)
	elif result=='r':
		x=True
		while x:
			filepath=input("Enter the path directory of the picture. Press q if same directory\n")
			if filepath=='q':
				x=False
				continue;
			try:
				os.chdir(filepath)
				x=False
			except FileNotFoundError:
				print("File not found. Please try again\n")
		while not x:
			filename=input("Enter the filename\n")
			train_image=cv2.imread(filename)
			if train_image is not None:
				x=True
				predict(train_image)
			else:
				continue;
	else:
		print(subjects)
		x=input("Do you want to save this classifier?(y/n)")
		if x=='y':
			classifier_name=input("What do want to name it? No need of extension\n")
			recog.write(classifier_name+".xml") #recog.save() if opencv 2._
			f=open(classifier_name+".txt","w+")
			j=0
			for i in subjects:
				if j==0:
					f.write(i)
					j+=1
					continue;
				f.write(","+i)
			f.close()
		break;


