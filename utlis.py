import cv2
import numpy as np

img= cv2.imread("images/17.jpeg")

import base64
encoded_string = base64.b64encode(img)
#print (encoded_string)
# decode frame
decoded_string = base64.b64decode(encoded_string)
decoded_img = np.fromstring(decoded_string, dtype=np.uint8)
decoded_img = decoded_img.reshape(img.shape)
# show decoded frame
cv2.imshow("decoded", decoded_img)

'''import base64
with open("images/17.jpeg", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
print(my_string)'''

#konturları bul
def Contours(img,minArea,filter,threshold=[100,100]):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, threshold[0], threshold[0])
    kernel = np.ones((5, 5))
    dial = cv2.dilate(canny, kernel, iterations=3)
    threshold = cv2.erode(dial, kernel,iterations=2)
    #threshold2 = cv2.resize(threshold,(0, 0), None, 0.4, 0.4)
    #cv2.imshow('threshold',threshold2)

    #dış konturlar.kare,
    contours, hiearchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fCont = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            epsilon = 0.051 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True) #köşe kontorl
            bbox = cv2.boundingRect(cnt)
            fCont.append([len(approx), area, approx, bbox, cnt])
    fCont = sorted(fCont, key = lambda x:x[1], reverse=True) #sırala
    return img, fCont


#Dosyanın köşelerini bul
#noktaları sırala
def sort_pts(edges):

    sorted_pts = np.zeros_like(edges)
    edges = edges.reshape((4, 2))
    sum = np.sum(edges, axis=1)
    sorted_pts[0] = edges[np.argmin(sum)]
    sorted_pts[3] = edges[np.argmax(sum)]
    diff = np.diff(edges, axis=1)
    sorted_pts[1]= edges[np.argmin(diff)]
    sorted_pts[2] = edges[np.argmax(diff)]

    return sorted_pts

#köşe noktalarıyla yeni çarpık görünütler
#pad ile kenarlara dolgu

def warping (img, points, w, h, pad=10):
    points =sort_pts(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(img, transformation_matrix, (w, h))
    imgWarp = warped_img[pad:warped_img.shape[0]-pad, pad:warped_img.shape[1]-pad]
    return imgWarp

img = cv2.resize(img,(0, 0), None, 0.4, 0.4)
cv2.imshow('Image', img)


while True:

    #filter=4 , area 50000 dosya
    _contours, conts = Contours(img, minArea=50000, filter=4)


    if len(conts) != 0: #listenin bos olmadıgından emin ol
        biggest = conts[0][2]

        imgWarp = warping(img, biggest, 711, 987)
        _contours, conts2 = Contours(imgWarp, minArea=2000, filter=4, threshold=[50, 50])


        if len(conts) != 0: #boyut
            for obj in conts2:
                #konturların yük ve gen bulmak için dik iki kenarı hesapla
                def Curves(x, y):
                    return np.sqrt(np.square(x[0]-y[0]) + np.square(x[1]-y[1]))

                cv2.polylines(_contours, [obj[2]], True, (255, 255, 0), 3)
                nEdges = sort_pts(obj[2])

                cv2.arrowedLine(_contours, (nEdges[0][0][0], nEdges[0][0][1]), (nEdges[1][0][0], nEdges[1][0][1]),
                                (0, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(_contours, (nEdges[0][0][0], nEdges[0][0][1]), (nEdges[2][0][0], nEdges[2][0][1]),
                                (0, 80, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(_contours, 'w:{}cm'.format(round((Curves(nEdges[0][0]//3, nEdges[1][0]//3)/10), 1)),
                            (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (0, 0, 255), 2)
                cv2.putText(_contours, 'h:{}cm'.format(round((Curves(nEdges[0][0]//3, nEdges[2][0]//3)/10), 1)),
                            (x - 110, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (0, 80, 255), 2)

        _contours = cv2.resize(_contours, (0, 0), None, 0.7, 0.7)
        cv2.imshow('Image2', _contours)

    cv2.waitKey(0)