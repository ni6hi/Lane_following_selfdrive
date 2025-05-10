import torch
import torchvision.transforms as transforms
import os
import cv2
import ssl
import torchvision
from pathlib import Path
import glob
import numpy as np
from collections import defaultdict

ssl._create_default_https_context = ssl._create_unverified_context


def select_device(device="", batch_size=None):

    return torch.device("cuda:0")


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # print(sem_img.shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return img, ratio, (dw, dh)


def detect(originalimg):
    # model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

    # Loading model:
    stride = 32
    device = torch.device("cuda:0")
    print(f"DEVICE IS {device}")
    model = torch.jit.load("/home/abhiyaan-nuc/yolopv2/data/weights/yolopv2.pt")
    half = True  # only if cuda is available, else set to false
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    img_size = 640
    # originalimg = cv2.imread('/home/mahesh/YOLOPv2/data/curvedlanes.jpeg')
    cv2.imshow("original image", originalimg)
    originalimg = cv2.resize(originalimg, (1280, 720), cv2.INTER_LINEAR)
    originalimg = letterbox(originalimg, img_size, stride=stride)[0]
    originalimg = originalimg[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    originalimg = np.ascontiguousarray(originalimg)
    originalimg = torch.from_numpy(originalimg).to(device)
    originalimg = (
        originalimg.half() if half else originalimg.float()
    )  # uint8 to fp16/32
    originalimg /= 255.0  # 0 - 255 to 0.0 - 1.0
    if originalimg.ndimension() == 3:
        originalimg = originalimg.unsqueeze(0)

    det_out, seg, ll_seg_out = model(originalimg)
    pad_h, pad_w, height, width = 12, 0, 372, 640
    ll_predict = ll_seg_out[:, :, 12:372, :]
    ll_seg_mask = torch.nn.functional.interpolate(
        ll_predict, scale_factor=int(2), mode="bilinear"
    )
    ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, scale_factor=2, mode="bilinear"
    )
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    # print(ll_seg_mask.shape)
    mask_img = np.zeros((da_seg_mask.shape[0], da_seg_mask.shape[1], 3), dtype=np.uint8)
    mask_img[ll_seg_mask == 1] = [255, 255, 255]
    gray_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("model output", gray_img)

    contourimg1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    contourimg2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    contourimg3 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    black_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    contours1, hierarchy = cv2.findContours(
        gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    maxcontour = contours1[0]
    secondcontour = contours1[0]
    thirdcontour = contours1[0]
    for contour in contours1:
        if len(contour) < len(maxcontour):
            thirdcontour = contour
    for contour in contours1:
        if len(contour) > len(maxcontour):
            maxcontour = contour
        if len(contour) < len(maxcontour) and len(contour) > len(secondcontour):
            secondcontour = contour
        elif (
            (len(contour) < len(maxcontour))
            and (len(contour) < len(secondcontour))
            and (len(contour) > len(thirdcontour))
        ):
            thirdcontour = contour

    contours = [maxcontour, secondcontour, thirdcontour]
    black_img = cv2.drawContours(black_img, contours1, -1, (0, 255, 0), 3)

    # curve fitting the contours -----------------------------------------------------------------------------------------------------------------------------------------------------------
    flag = 0

    for contour in contours:

        if flag == 0:
            print("first contour")
        elif flag == 1:
            print("second contour")
        elif flag == 2:
            print("third contour")
            print(contour)

        contourxpoints = []
        contourypoints = []

        for point in contour:
            x = point[0][0]
            y = point[0][1]
            contourxpoints.append(x)
            contourypoints.append(y)

        deg = 2
        curvefit = np.polyfit(contourxpoints, contourypoints, deg)
        # print(curvefit)
        x = np.linspace(0, 1920, 1920)
        # print((xpoints, ypoints))
        minx = min(contourxpoints)
        maxx = max(contourxpoints)
        # print(minx, maxx)
        for point in x:
            if (point < maxx) and (point > minx):
                y = curvefit[deg]
                for n in range(deg):
                    y = y + ((curvefit[n]) * (point ** (deg - n)))
                y = int(y)
                x = int(point)
                if y >= 0:
                    if flag == 0:
                        contourimg1 = cv2.circle(
                            contourimg1,
                            (x, y),
                            radius=15,
                            color=(255, 255, 255),
                            thickness=2,
                        )
                    elif flag == 1:
                        contourimg2 = cv2.circle(
                            contourimg2,
                            (x, y),
                            radius=15,
                            color=(255, 255, 255),
                            thickness=2,
                        )
                    elif flag == 2:
                        contourimg3 = cv2.circle(
                            contourimg3,
                            (x, y),
                            radius=15,
                            color=(255, 255, 255),
                            thickness=2,
                        )

        flag += 1

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    contourimg1 = cv2.cvtColor(contourimg1, cv2.COLOR_BGR2GRAY)
    # contourimg2 = cv2.cvtColor(contourimg2, cv2.COLOR_BGR2GRAY)
    # contourimg3 = cv2.cvtColor(contourimg3, cv2.COLOR_BGR2GRAY)
    # final_lanes = cv2.bitwise_or(contourimg1, contourimg2)
    # final_lanes = cv2.bitwise_or(contourimg3, final_lanes)
    final_lanes = contourimg1
    cv2.imshow("final lanes", final_lanes)
    cv2.imshow("contours", black_img)
    cv2.waitKey(1)
