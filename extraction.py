import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img=cv2.imread("images/scanned_note.jpg")
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(640,480))

hImg,wImg,_=img.shape
fichier=open("extacted_note1.txt","w")
fichier.write(pytesseract.image_to_string(img))
fichier.close()


cv2.imshow('result',img)
cv2.waitKey(0)