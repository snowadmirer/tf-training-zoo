#coding=utf-8
import random
import cv2
def random_aspect_ratio(image, bboxes=None, min_ratio=0.8, max_ratio=1.2, flags=cv2.INTER_NEAREST):
    if random.uniform(0.0, 1.0) < 0.5:
        return image, bboxes

    height, width = image.shape[:2]
    ratio = random.uniform(min_ratio, max_ratio)
    if random.uniform(0.0, 1.0) < 0.5:
        height = int(height * ratio)
        if bboxes is not None:
            bboxes[:,[1,3]] = bboxes[:,[1,3]] * ratio
    else:
        width = int(width * ratio)
        if bboxes is not None:
            bboxes[:,[0,2]] = bboxes[:,[0,2]] * ratio
    image = cv2.resize(image, (width, height), flags)

    return image, bboxes
