import cv2
import numpy as np

def get3Dpoint(u, v, A, rvec, tvec, s):
    # Generamos el vector m
    uv = np.array([[u, v, 1]], dtype=np.float).T

    # Obtenemos R a partir de rvec
    R, _ = cv2.Rodrigues(rvec)
    Inv_R = np.linalg.inv(R)

    # Parte izquierda m*A^(-1)*R^(-1)
    Izda = Inv_R.dot(np.linalg.inv(A).dot(uv))

    # Parte derecha
    Drch = Inv_R.dot(tvec)

    # Calculamos S porque sabemos Z = 0
    s = 0 + Drch[2][0] / Izda[2][0]

    XYZ = Inv_R.dot(s * np.linalg.inv(A).dot(uv) - tvec)

    return XYZ


def get2Dpoint(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        uvpoint.clear()
        uvpoint.append(x)
        uvpoint.append(y)
        cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
        cv2.imshow("Original Image", img)
        print("UV point: \n")
        print(uvpoint)
