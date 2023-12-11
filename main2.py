#IMPORTS
import matplotlib.pyplot as plt
from transformations import *
from drawBox2 import drawBox

#DEFINITION OF SOME FUNCTIONS THAT USE VARIABLES FROM THIS CLASS
def get2D3Dcoordinates(event,x_i,y_i,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        imagearray.append([x_i,y_i])
        cv2.circle(img,(x_i,y_i),10,(0,255,0),cv2.FILLED)
        cv2.imshow("Original Image", img)
        x_r = float(input("\nEnter any value of real world X coordinate: "))
        y_r = float(input("\nEnter any value of real world Y coordinate: "))
        #z_r = float(input("\nEnter any value of real world Z coordinate: "))
        z_r = float(0.0)
        worldarray.append([x_r,y_r,z_r])
        print("Image coordinates array: \n")
        print(imagearray)
        print("\n \n")
        print("Real world coordinates array:  \n")
        print(worldarray)

def detectPerson(img):
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutput = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layersOutput:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.95:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    boxeses = []
    if len(indexes) and len(boxes) > 0:
        counting = 0
        for i in indexes.flatten():
            boxeses.append(boxes[i])
            counting = counting + 1

    bbox1 = []
    bbox2 = []
    if (len(boxes) > 0) and len(indexes) >= 2:
        bbox1 = boxeses[0]
        bbox2 = boxeses[1]
        print(bbox1)
        print(bbox2)

    return bbox1, bbox2

def updateHeatMap(x_real, y_real):
    x = round(x_real/100)
    y = round(y_real/100)
    if x < int(length/100) and y < int(width/100):
        heatMap[y][x] = heatMap[y][x] + 1


#INITIALIZATION OF VARIABLES
imagearray = []
worldarray = []
finalpoints = []
s = 0
z = 0
uvpoint = []
counter_detect = 0
counter = 0

#INITIALIZATION OF VIDEO CAPTURING
cap = cv2.VideoCapture("videos/Case3.mov")
#cap = cv2.VideoCapture("http://172.20.10.5:8080/video")
#cap = cv2.VideoCapture("http://192.168.0.223/video.mjpg")

#DEFINITION OF WIDTH AND LENGTH OF THE RECTANGLE
length = float(input("\nEnter length value: "))
width = float(input("\nEnter width value: "))

#DEFINITION OF THE POINTS FOR EXTRINSIC CALIBRATION
success, img = cap.read()
cv2.imshow("Original Image", img)
cv2.setMouseCallback("Original Image", get2D3Dcoordinates)
cv2.waitKey(0)
cv2.destroyWindow("Original Image")

# INITIALIZATION OF YOLOV3 NEURAL NETWORK
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# INITIALIZATION OF SVM+HOG DETECTOR
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#INITIALIZATION OF TRACKER
tracker1 = cv2.legacy.TrackerCSRT_create()
tracker2 = cv2.legacy.TrackerCSRT_create()
#tracker1 = cv2.legacy_TrackerMOSSE.create()
#tracker1 = cv2.legacy_TrackerKCF.create()

#INITIALIZATION OF HEATMAP
heatMap = np.zeros((int(width/100), int(length/100)))

#INITIALIZATION OF INTRINSIC PARAMETERS OBTAINED FROM CAMERA CALIBRATION
#puntos3d = np.array([[0, 0, 0],
#                    [0, 50, 0],
#                    [100, 50, 0],
#                    [100, 0, 0]], dtype=np.float32)

puntos3d = np.array(worldarray,dtype=np.float32)

#puntos2d = np.array([[164, 1002],
#                     [555, 637],
#                     [1390, 707],
#                     [1377, 1204]], dtype=np.float32)

puntos2d = np.array(imagearray, dtype=np.float32)

#Normal camera
#camera_matrix = np.array([[3.03934002e+03, 0.00000000e+00, 1.98751095e+03],
#                          [0.00000000e+00, 3.03193288e+03, 1.52062514e+03],
#                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float64)

#720
#camera_matrix = np.array([[1.08812596e+03, 0.00000000e+00, 6.24065214e+02],
#                          [0.00000000e+00, 1.08556623e+03, 3.66984576e+02],
#                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float64)

#DJI
camera_matrix = np.array([[923.80729027, 0.00000000e+00, 649.6620753],
                          [0.00000000e+00, 921.59308316, 346.45607893],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float64)

#distortion_vector = np.array((2.82851714e-01, -1.44902310e+00,  3.57423197e-03, -6.62060855e-04, 2.21601970e+00))
#distortion_vector = np.array((2.61953814e-01, -8.23453706e-01, 6.02116664e-03, -5.57801916e-04, -6.85127579e-01))
distortion_vector = np.array((3.49928316e-01, -2.59878576e+00, 5.66083119e-03, 6.11535112e-03, 6.30730568e+00))

#OBTAINING ROTATION AND TRANSLATION VECTORS FROM INTRINSIC PARAMETERS AND REAL TIME CALIBRATION POINTS
ret, rvec, tvec = cv2.solvePnP(puntos3d , puntos2d, camera_matrix, distortion_vector)

#INITIALIZATION OF THE BOUNDING-BOX USING YOLOV3 PERSON DETECTION
#bbox1 = cv2.selectROI("Tracking", img, False)
bbox1, bbox2 = detectPerson(img)
tracker1.init(img, bbox1)
tracker2.init(img, bbox2)

#INITIALIZATION OF THE BOUNDING-BOX USING HOG PERSON DETECTION
#(rects, weights) = hog.detectMultiScale(img, winStride=(4, 4),
#	padding=(0, 0), scale=1.05)
#if len(rects) > 0:
#    bbox1 = rects[1]
#    tracker1.init(img, bbox1)

#INITIALAZING POSITION GRAPHIC
plt.ion()
figA, ax = plt.subplots()
figC, ax2= plt.subplots()
background = plt.imread("images/background2.png")
ax.imshow(background, extent=[0, length, 0, width])
ax2.imshow(background, extent=[0, length, 0, width])
plot1 = ax.scatter([], [])
plot2 = ax2.scatter([], [])
ax.set_xlim(0, length)
ax.set_ylim(0, width)
ax2.set_xlim(0, length)
ax2.set_ylim(0, width)
plt.axis('scaled')

#INITIALIZING HEATMAP
figB, ax1 = plt.subplots()
ax1.set_xlim(0, length/100)
ax1.set_ylim(0, width/100)
plt.axis('scaled')


while True:

    #READING NEW FRAME
    timer = cv2.getTickCount()
    success_tracker1, bbox1 = tracker1.update(img)
    success_tracker2, bbox2 = tracker2.update(img)
    if not success:
        break
    #img = imutils.resize(img, width=min(720, img.shape[1]))
    success_tracker1, bbox1 = tracker1.update(img)

    #DRAWING BOX IF DETECTED
    if success_tracker1 and success_tracker2:
        drawBox(img, bbox1, bbox2)
    else:
        # cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
        if success:
            bbox1, bbox2 = detectPerson(img)
        if (len(bbox1) > 0 or len(bbox2) > 0):
            tracker1.init(img, bbox1)
            tracker2.init(img, bbox2)

    #PRINT FRAME WITH FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
    if success:
        cv2.imshow("Tracking", img)

    #MAKE THE TRANSFORMATION FROM 2D TO 3D EVERY 6 FRAMES
    if counter == 6:
        counter = 0
        u = int(bbox1[0]) + int(bbox1[2]) / 2
        v = int(bbox1[1]) + int(bbox1[3])

        XYZ = get3Dpoint(u, v, camera_matrix, rvec, tvec, s)

        updateHeatMap(XYZ[0][0], XYZ[1][0])

        u2 = int(bbox2[0]) + int(bbox2[2]) / 2
        v2 = int(bbox2[1]) + int(bbox2[3])

        XYZ2 = get3Dpoint(u2, v2, camera_matrix, rvec, tvec, s)

        updateHeatMap(XYZ2[0][0], XYZ2[1][0])

        plot1.set_offsets((XYZ[0][0], XYZ[1][0]))
        plot2.set_offsets((XYZ2[0][0], XYZ2[1][0]))
        if XYZ[0][0] < 500 and XYZ[1][0] > 500:
            plot1.set_color("red")
        else:
            plot1.set_color("blue")

        if XYZ[0][0] < 500 and XYZ[1][0] > 500:
            plot2.set_color("red")
        else:
            plot2.set_color("blue")
        figA.canvas.draw()
        figC.canvas.draw()

        #ax1.imshow(heatMap, cmap='YlOrBr', interpolation='nearest')
        #plt.show()

    #MAKE THE PERSON DETECTION EVERY 30 FRAMES
    if counter_detect == 30:
        counter_detect = 0
        if success:
            bbox1, bbox2 = detectPerson(img)
        if (len(bbox1) > 0 or len(bbox2) > 0):
            tracker1.init(img, bbox1)
            tracker2.init(img, bbox2)

    counter_detect = counter_detect + 1
    counter = counter + 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

ax1.imshow(heatMap, cmap='YlOrBr', interpolation='nearest')
plt.show()
cv2.waitKey(0)