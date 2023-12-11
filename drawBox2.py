import cv2

def drawBox(img, bbox1, bbox2):
    x, y, w, h = int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255),1,1)
    x, y, w, h = int(bbox2[0]), int(bbox2[1]), int(bbox2[2]), int(bbox2[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 1, 1)