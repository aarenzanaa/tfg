import cv2

def drawBox(img, bbox1):
    x, y, w, h = int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255),1,1)
    cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)