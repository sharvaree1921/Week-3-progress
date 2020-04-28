# Week-3,4,5-progress
Documentation of what I have learned in the week 7th-12th,13th-19th,20th-26th April

## OpenCV
OpenCV is open source **computer vision** library.It supports C,C++,python,java on windows,max,Linux operating systems.
(_comuter vision is the way we teach intelligence to machines and making them see things just like humans.ex-self driving car_)
OpenCV helps in analyzing and manipulating images,videos,etc. through its inbuilt functions with help of numpy and matplotlib.

#### Two types of digital images-
1. Gray scale image(2D,gives light intensity at each pixel,one channel)
2. Coloured image(3D,3rd dimension is of colour,three channels namely R,G,B)

I have installed OpenCV's 4.2.0 version on my Ubuntu 18.04 with python version 3.6.9.
OpenCV stores images in form of B,G,R format but not in R,G,B format.

### Image Processing using opencv
1. How to read,write images and videos from camera.
   ```
      cv2.imread()
      cv2.VideoCapture()
      cv2.videoWriter()
      cv2.imshow()
   ```
2. Drawing Geometric Shapes on images
   ```
      cv2.line(img,l_b,u_b,color,thickness)
      cv2.arrowedLine(img,l_b,u_b,color,thickness)
      cv2.rectangle(img,pt1,pt2,color,thickness)
      cv2.circle(img,center,radius,color,thickness)
      cv2.putText(img,text,org,font_face,font_size,font_color,thickness,line_type)
   ```
3. Setting camera parametres
4. Show date and time on videos
   ```
      import datatime
      datet=str(datatime.datatime.now())
      #then use cv2.putText to print
   ```
5. Handling mouse events
6. Using split,merge,resize,add,addweighted,ROI
   ```
      cv2.split() #spliting b,g,r channels of coloured images
      cv2.merge() #merging b,g,r channels of splitted image
      cv2.add(img1,img2)
      cv2.resize(src,size)
      cv2.addWeighted(img1,wt1,img2,wt2,const)
   ```
7. Bitwise Operators
   ```
      cv2.bitwise_and(img1,img2)
      cv2.bitwise_or(img1,img2)
      cv2.bitwise_not(img)
      cv2.bitwise_xor(img1,img2)
   ```
8. Binding trackbars 
   ```
      cv2.namedWindow()
      cv2.creatTrackbar(trackbar_name,img,lower_b,upper_b,function)
      cv2.getTrackbarPos(trackbar_name,img)
   ```
9. Object Detecion and object tracking using hsv color space
   ```
      l_b=np.array([l_h,l_s,l_v])
      u_b=np.array([u_h,u_s,u_v])
      mask=cv2.inRange(img,l_b,u_b)
      res=cv2.bitwise_and(frame,frame,mask=mask)
   ```
10. Simple Image Thresholding
   ```
      _,th1=cv2.threshold(img,threshold_value,max_value,threshold_type)
      #threshold types-cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV
   ```
11. Adaptive Thresholding
   ```
      #thresholding for smaller regions
      cv2.adaptiveThreshold(src,max_value,adaptive_method,threshold_type,blocksize,const)
   ```
12. Matplotlib using opencv
13. Morpholigical Transformations
   ```
      cv2.dilate(mask,kernel,iteration)
      cv2.erode(mask,kernel,iteration)
      cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)  #erosion followed by dilation 
      cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel) #dilation followed by erosion
   ```
14. Smoothing and Blurring images
   ```
      #homogenous filter,gaussian filter,mean filter,bilateral filter
      cv2.filter2D(img,depth,kernel)
      cv2.blur(img,kernel_size)
      cv2.GaussianBlur(img,kernel_size,sigma_x_value)
      cv2.medianBlur(img,kernel_size)
   ```
15. Image Gradients and edge detection
   ```
      cv2.Laplacian(img,datatype,kernel_size)
      cv2.Sobel(img,datatype,1,0)  #sobel x
      cv2.Sobel(img,datatype,0,1)  #sobel y
   ```
16. Canny Edge Detection
   ```
      cv2.Canny(img,thresh1,thresh2)  #two threshold values for last step hysterisis
   ```
17. Image Pyramids
   ```
      #two kinds of image pyramids-gaussian and laplacian
      cv2.pyrDown(img)
      cv2.pyrUp(img)
   ```
18. Image Blending
   ```
      #load the two desired images
      #Form the Gaussian pyramids of two images upto some level,say 6.
      #From Gaussian,form Laplacian pyramids
      #Join the right and left half of each level of Laplacian pyramids
      #Reconstruct the original image
   ```
19. Finding and Drawing Contours
   ```
      contours,hierarchy=cv2.FindContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      cv2.drawContours(img,contours,index,color,thickness)
   ```
20. Motion Detection and Tracking
21. Detect Simple Geometric shapes
   ```
      approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
      cv2.boundingRect(approx)
   ```
22.Understanding Image Histograms
   ```
      b,g,r=cv2.cv2.split(img)
      plt.hist(b.ravel(),256,[0,256])
      plt.hist(g.ravel(),256,[0,256])
      plt.hist(r.ravel(),256,[0,256])
   ```
23. Template Matching
   ```
      w, h = template.shape[::-1]  
      res = cv2.matchTemplate(grey_img, template, cv2.TM_CCORR_NORMED )
   ```
24. Hough Line Transform using Hough Line Method
   [Understand here](https://github.com/MananKGarg/SOC_20_Virtual_Keyboard/blob/master/SoC_OpenCV-master/29.%20%5BAnkit   %5D%20Hough%20Line%20Transform%20using%20HoughLines%20method%20.md)
25. Probabilistic Hough Line Transform using Hough Line Method
   [Understand here](https://github.com/MananKGarg/SOC_20_Virtual_Keyboard/blob/master/SoC_OpenCV-master/30.%20%5BAnkit%5D%20Probabilistic%20Hough%20Transform%20using%20HoughLinesP.md)
26. Road Lane Line Detection
27. Circle Detection
   ```
      cv2.HoughCircles(img,detection method,dp,min_dist,parameter1,parameter2,minRadius,maxRadius)
      detected_circles = np.uint16(np.around(circles))
   ```
28. Face Detection
   ```
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      faces = face_cascade.detectMultiScale(gray, scale_factor,min_neighbours)
   ```
29. Eye Detection
   ```
      eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml') 
      eyes = eye_cascade.detectMultiScale(roi_gray)
   ```
30. Detect Corners using Harris Corner Detect method
   ```
      cv2.cornerHarris(img,block_size,k_size,harris_param)
   ```
31. Detect Corners using shi Tomasi method
   ```
      corners = cv.goodFeaturesToTrack(img,no_of_corners,quality,Euclidean_distance)
   ```
32. Background subtraction 
      [Understand Here](https://www.youtube.com/watch?v=eZ2kDurOodI&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=43)

References-
[OpenCV_GitHub](https://github.com/MananKGarg/SOC_20_Virtual_Keyboard/tree/master/SoC_OpenCV-master)
[OpenCV_youtube](https://www.youtube.com/playlist?list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K)
      
