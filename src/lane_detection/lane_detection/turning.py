import cv2
import numpy as np
import math

MASK_THRESHOLD = 3.5/8

def dist(x, y, p1, p2):
    d_x = (p1[0]-p2[0])
    d_y = (p1[1]-p2[1])
    c = p1[0]*d_y - p1[1]*d_x
    dist = y*d_x - x*d_y + c
    denom = d_x*d_x + d_y*d_y
    denom = denom**0.5
    return abs(dist)/denom

def perpdist(x, y, slope, intercept):
    num = abs(slope*x - y + intercept)
    denom =(slope*2 + 1)*0.5
    return(num/denom)

def get_cat(line):
    p1 = [line[0][0], line[0][1]]
    p2 = [line[0][2], line[0][3]]
    slope = math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    if(slope < 0):
        slope += math.pi
    if(slope > math.pi):
        slope -= math.pi

    return((slope, p1[1] - math.tan(slope)*p1[0], p1, p2))

def main():
    img1 = cv2.imread("int_top.png")
    cv2.imshow("original image", img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.inRange(img1, 130, 230)

    mask = np.zeros(img1.shape[:2], np.uint8)
    mask[0:int(MASK_THRESHOLD*img1.shape[1]), 0:img1.shape[1]] = 255
    img1 = cv2.bitwise_and(img1, img1, mask=mask)
    img1 = cv2.blur(img1, (3,3))
    img1 = cv2.medianBlur(img1, 3)

    w = img1.shape[1] #len(img1)
    h = img1.shape[0] #len(img1[0])
    cX = h//2
    cY = w//2

    M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
    img = cv2.warpAffine(img1, M, (w, h), borderValue=(255,255,255))
    cv2.imshow("abs", img)

    # finding lines with probablistic hough transform

    # binaryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaryimg = img1
    canny = cv2.Canny(binaryimg, 50, 200, None, 3)
    cdst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLinesP(canny, 0.5, np.pi / 180, 10, None, 50, 10)
    categories = []

    # adding the first line to the clusters list
    line = lines[0]
    categories.append(get_cat(line))

    # clustering the remaining lines
    for line in lines:
        line_info = get_cat(line)
        slope = line_info[0]
        p1 = line_info[2]
        p2 = line_info[3]

        thresh = 10
        flag = False
        for cat in categories:
            if(abs(slope - cat[0])<0.5 or abs(slope - math.pi - cat[0])<0.1 or abs(slope + math.pi - cat[0])<0.1) and (dist(p1[0], p1[1], cat[2], cat[3]) < thresh):
                flag = True
                break
        if(flag):
            continue
        else:
            categories.append(get_cat(line))

    whitepoints = cv2.findNonZero(canny)

    lines2 = np.array([[math.tan(line[0]), line[1], line[2][0], line[2][1], line[3][0], line[3][1]] for line in categories])

    # checking for 45 deg

    s1 = lines2[0][0]
    if (abs(s1) - 1 < 0.1):
        logic = 0
    else:
        logic = 1

    # figuring out which line is horizontal and which is vertical

    horiz = []
    vert = []
    for line in lines2:
        slope = line[0]
        if (logic):
            if abs(slope) < 1:
                horiz.append(line)
            else:
                vert.append(line)
        else:
            if (slope < 0):
                horiz.append(line)
            else:
                vert.append(line)

    if (len(vert) < len(horiz)):
        vert, horiz = horiz, vert
    newhoriz = []
    for line in horiz:
        if (line[0] - vert[0][0] > 0.3):
            newhoriz.append(line)


    horiz = newhoriz
    if (len(horiz)>0):

        # finding bounds of the horizontal line
        minpoint = [horiz[0][2], horiz[0][3]]
        maxpoint = [horiz[0][4], horiz[0][5]]

        if len(whitepoints) > 0 and len(horiz) > 0 and len(vert) > 0:
            mindist = 100000
            maxdist = 0
            for point in whitepoints:
                point = point[0]
                d1 = perpdist(point[0], point[1], horiz[0][0], horiz[0][1])
                if (d1 < 10):
                    d2 = perpdist(point[0], point[1], vert[0][0], vert[0][1])
                    if(d2 < mindist):
                        mindist = d2
                        minpoint = [point[0], point[1]]
                    if(d2 > maxdist):
                        maxdist = d2
                        maxpoint = [point[0], point[1]]

    # drawing the lines


    # drawing the vertical lines
    for line in vert:
        slope = line[0]
        intercept = line[1]
        y3 = int(intercept)
        y4 = int(slope*cdst.shape[1] + intercept)
        if (abs(y3) > 10000):
            y3 = 10000
        if (abs(y4) > 10000):
            y4 = 10000
        cv2.line(cdst, (int(0), y3), (cdst.shape[1], y4), (255,255,0), 3, cv2.LINE_AA)

    # drawing the horizontal lines
    for i,line in enumerate(horiz):
        slope = line[0]
        intercept = line[1]
        coords = []
        coords.append(int((minpoint[1] - intercept)/slope))
        coords.append(minpoint[1])
        coords.append(int((maxpoint[1] - intercept)/slope))
        coords.append(maxpoint[1])

        for i in range(len(coords)):
            point = coords[i]
            if (abs(point) > 10000):
                coords[i] = int(abs(point)/point*10000)
        cv2.line(cdst, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0,255,255), 3, cv2.LINE_AA)


    cv2.imshow('lanes', cdst)
    cv2.waitKey()

if __name__ == "__main__":
    main()
