import cv2
import numpy as np

hand_cascade = cv2.CascadeClassifier('palm.xml')
hand_cascade1 = cv2.CascadeClassifier('fist.xml')
hand_cascade2 = cv2.CascadeClassifier('aGest.xml')
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hand = hand_cascade.detectMultiScale(gray, 1.4, 5)
	hand1 = hand_cascade1.detectMultiScale(gray, 1.4, 5)
	hand2 = hand_cascade2.detectMultiScale(gray, 1.4, 5)
	
	for (x, y, w, h) in hand:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
		cv2.putText(img,"There is a Palm", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0))
		
	for (x, y, w, h) in hand1:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
		cv2.putText(img,"There is a Fist", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0))
	
	for (x, y, w, h) in hand2:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
		
	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		
cap.release()
cv2.destroyAllWindows()