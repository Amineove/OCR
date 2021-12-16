import cv2

img=cv2.imread("images/20200601_132003.jpg")
scale_percent=0.2
width=int(img.shape[1]*scale_percent)
height=int(img.shape[0]*scale_percent)
dim=(width,height)

resized=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
cv2.imshow("Resized",resized)
cv2.imshow("original",img)
cv2.imwrite("images/cin_res.jpg",resized)
cv2.waitKey(0)