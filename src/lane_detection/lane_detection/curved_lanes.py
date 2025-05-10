import cv2
import numpy as np

DIST_THRESHOLD = 300 # Distance threshold between consecutive dashes
MASK_THRESHOLD = 4.5/8 # Fraction of the image to mask out
PARAM_THRESHOLD = 0.4 # Difference in the (ymax-ymin/xmax-xmin) of contours (to eliminate the horizontal line)
DASH_SIZE_LOW = 30 # Minimum size to be considered a dashed lane
DASH_SIZE_HIGH = 600 # Maximum size to be considered a dashed lane

# Finding ymax - ymin/xmax - xmin for a contour

def find_param(contour):
    bottom = contour[contour[:, :, 1].argmax()][0][1]
    top = contour[contour[:, :, 1].argmin()][0][1]
    left = contour[contour[:, :, 0].argmin()][0][0]
    right = contour[contour[:, :, 0].argmax()][0][0]

    return abs(bottom - top)/abs(left - right)

def main():
    img = cv2.imread("lanethresh.png")
    binaryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #binaryimg = cv2.inRange(binaryimg, 170, 255)

    # Pre processing the image to get better results

    mask = np.zeros(binaryimg.shape[:2], np.uint8)
    mask[int(MASK_THRESHOLD*binaryimg.shape[1]):binaryimg.shape[1], 0:binaryimg.shape[0]] = 255
    binaryimg = cv2.bitwise_and(binaryimg, binaryimg, mask=mask)
    binaryimg = cv2.blur(binaryimg, (3,3))
    binaryimg = cv2.medianBlur(binaryimg, 3)
    cv2.imshow("threshold", binaryimg)

    # Finding contours

    blackimg = np.zeros((binaryimg.shape[0], binaryimg.shape[1], 3), dtype=np.uint8)

    contours1, _ = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    newcontours = []
    contours = [contour for contour in contours1]
    solidlane = contours[0]


    # getting biggest contour - which corresponds to solid lane

    for contour in contours:
        if len(contour) > len(solidlane):
            solidlane = contour

    firstdash = solidlane
    thirdcontour = solidlane

    # getting the closest dashed line
    # TODO: on NUC change botpoint to be max X coordinate, not Y coordinate
    xmax = 0
    for contour in contours:
        if len(contour) < len(solidlane) and len(contour) < DASH_SIZE_HIGH and len(contour) > DASH_SIZE_LOW:
            botpoint = contour[contour[:, :, 1].argmax()][0]
            if (botpoint[1] > xmax):
                xmax = botpoint[1]
                firstdash = contour

    # finding ymax - ymin / xmax - xmin for the first dash

    param = find_param(firstdash)

    # getting the remaining dashes


    dashedlines = [firstdash]
    lastdash = firstdash
    for contour in contours:
        if (DASH_SIZE_LOW < contour.shape[0]) and (contour.shape[0] < DASH_SIZE_HIGH):
            M1 = cv2.moments(contour)
            centre1x = int(M1['m10']/M1['m00'])
            centre1y = int(M1['m01']/M1['m00'])
            M2 = cv2.moments(lastdash)
            centre2x = int(M2['m10']/M2['m00'])
            centre2y = int(M2['m01']/M2['m00'])

            param2  = find_param(contour)

            dist = ((centre2y - centre1y)**2 + (centre2x - centre1x)**2)**0.5
            if (dist < DIST_THRESHOLD) and (abs(param2 - param) < PARAM_THRESHOLD):
                dashedlines.append(contour)
                lastdash = contour
            else:
                newcontours.append(contour)

    # Uncomment to clear all the found contours
    contours = newcontours

    # TODO: clearing the found contours from the list
    testimg = np.zeros((binaryimg.shape[0], binaryimg.shape[1], 3), dtype=np.uint8)
    if len(newcontours) != 0:
        testimg = cv2.drawContours(testimg, newcontours, -1, (255, 255, 255), 3)

    if (len(dashedlines) != 0):
        laneslist = [dashedlines, [solidlane]]
    else:
        laneslist = [[solidlane]]


    # curve fitting the lanes

    finallanes = np.zeros((binaryimg.shape[0], binaryimg.shape[1], 3), dtype=np.uint8)
    print(len(dashedlines))
    deg = 2
    for i, list in enumerate(laneslist):
        xpoints = []
        ypoints = []
        for j, contour in enumerate(list):
            bottom = contour[contour[:, :, 1].argmax()][0]

            for point in contour:
                if (bottom[1] > 0.9*binaryimg.shape[1]):
                    if (point[0][1] > (5/8)*binaryimg.shape[1]):
                        xpoints.append(point[0][0])
                        ypoints.append(point[0][1])
                else:
                    xpoints.append(point[0][0])
                    ypoints.append(point[0][1])

        coeffs = np.polyfit(ypoints, xpoints, deg)

        for y in range(min(ypoints), max(ypoints)):
            x = coeffs[deg]
            for n in range(deg):
                x = x + ((coeffs[n])*(y**(deg - n)))
            x = int(x)
            y = int(y)
            if i == 0:
                finallanes = cv2.circle(finallanes, (x, y), radius=5, color=(255, 255, 255), thickness=-1)
            else:
                finallanes = cv2.circle(finallanes, (x, y), radius=5, color=(255, 0, 0), thickness=-1)


    # visualizing lanes

    blackimg = cv2.drawContours(blackimg, dashedlines, -1, (255, 0, 0), 3)
    blackimg = cv2.drawContours(blackimg, [solidlane], -1, (0, 0, 255), 3)
    cv2.imshow("contours", blackimg)
    cv2.imshow("final lanes", finallanes)
    #cv2.imshow("remaining contours", testimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
