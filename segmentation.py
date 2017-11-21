import cv2
def segmentation(img):
    #location
    imageLoc = img

    #read image
    img = cv2.imread(imageLoc , 1)

    #OpenCV represents RGB images as multi-dimensional NumPy arrays…but in reverse order!
    #This means that images are actually represented in BGR order rather than RGB!

    #Plotting numpy arrays as images
    #pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #define kernel for morphological
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    #grayscale
    gray = cv2.imread(imageLoc,0)

    #morphological gradient
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    #pyplot.imshow(gradient,cmap='gray')
    #plt.show()
    #get the threshold of morphological gradient
    retval,out = cv2.threshold(gradient,50,200,cv2.THRESH_BINARY)

    #apply dilation to output so broken pixels are fixed
    out = cv2.dilate(out,kernel,iterations=1)

    #pyplot.imshow(out,cmap='gray')
    #pyplot.show()

    #
    #find obj
    rects, contours, h = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #duplicate to test detection
    imgCopy=img

    #define min and max area for objects
    min_a = .001
    max_a = 1
    #
    segm = []
    #draw rectangle and add object in segm array as individual obj
    #print("contours detected:",len(contours))
    obj = 0
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        a_ratio = (w*h) / (imgCopy.shape[0] * imgCopy.shape[1])
        if (a_ratio >= min_a and a_ratio <= max_a):
            obj+=1
            segm.append(tuple([x,y,w,h]))
            imgCopy = cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,255,0),10)

    #print("Objects detected:",obj)
    return segm,img,out



