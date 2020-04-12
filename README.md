# Week-3-progress
Documentation of what I have learned in the week 7th-12th April

## OpenCV
OpenCV is open source **computer vision** library.It supports C,C++,python,java on windows,max,Linux operating systems.
(_comuter vision is the way we teach intelligence to machines and making them see things just like humans.ex-self driving car_)
OpenCV helps in analyzing and manipulating images,videos,etc. through its inbuilt functions with help of numpy and matplotlib.

#### Two types of digital images-
1. Gray scale image(2D,gives light intensity at each pixel,one channel)
2. Coloured image(3D,3rd dimension is of colour,three channels namely R,G,B)

I have installed OpenCV's 4.2.0 version on my Ubuntu 18.04 with python version 3.6.9.

### Introduction to OpenCV-
1. Reading an image.
     >cv2.imread('_image.ext_')
2. Writing an image.
     >cv2.imwrite()
3. Displaying an image.
     >cv2.imshow('_windowname_',_image_)
4. Saving an image.
5. Extracting RGB values of a pixel.
     >cv2.split(_image_)
6. Resizing an image.
     >cv2.resize(_image,(w,h)_)
7. Rotating an image.
     >cv2.getRotationMatrix2D(_center,angle,scale_)
     
     >cv2.warpAffine(_image,matrix,(w,h))
8. Color Space(_RGB_,_CMYK_,_HSV_)
9. Arithmetic Operations
     >cv2.addWeighted(_img1,w1,img2,w2,gamma)
     
     >cv2.subtract(_img1,img2)
10. Bitwise Operations.
     >cv2.bitwise_and(img1,img2)
     
     >cv2.bitwise_or(img1,img2)
     
     >cv2.bitwise_xor(img1,img2)
     
     >cv2.bitwise_not(img)

### Image Processing
1. Image resizing
2. Image Erosion(Structuring element,kernel is required.Erodes the values)
3. Blurring Image
4. Making border around images
5. Grayscaling
6. Canny Edge Detection
7. Erosion and Dilation of images(Erosion followed by Dilation helps in reducing noise)
8. 
