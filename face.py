import cv2

def detect_faces(cascade, image, scaleFactor = 1.1):
	# Making a copy of image
	image_copy = image.copy()          
	
	# Converting to gray image
	gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)         

	# Detect faces using OpenCV Cascades
	faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

	# Plotting rectangle over faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              

	return image_copy

haar_face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')
video = cv2.VideoCapture(0)
while True:
	ret, frame = video.read()
	face_detected = detect_faces(haar_face_cascade, frame)
	cv2.imshow('face', face_detected)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 
video.release()
cv2.destroyAllWindows()