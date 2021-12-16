import cv2
import numpy as np
import pytesseract
originalImage=cv2.imread("images/scanned_note.jpg")
pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
originalImage=cv2.resize(originalImage,(640,480))
#blur = cv2.blur(originalImage,(5,5))

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imwrite("images/nscann.jpg",blackAndWhiteImage)
hImg,wImg,_=originalImage.shape
boxes=pytesseract.image_to_data(blackAndWhiteImage)

for a, b in enumerate(boxes.splitlines()):
    print(b)
    if a != 0:
        b = b.split()
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            #cv2.putText(blackAndWhiteImage, b[11], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(blackAndWhiteImage, (x, y), (x + w, y + h), (255, 255, 255), 1)

cv2.imshow('img', blackAndWhiteImage)



#cv2.imshow('Original image',originalImage)
cv2.imshow('Gray image', grayImage)
#cv2.imshow('blur',blur)
cv2.waitKey(0)