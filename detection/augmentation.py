#coding=utf-8
import random
import cv2
import numpy as np

def random_aspect_ratio(image, bboxes=None, min_ratio=0.8, max_ratio=1.2, flags=cv2.INTER_NEAREST, random_rate=0.5):
    if random.uniform(0.0, 1.0) < random_rate:
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

def random_pad(image, bboxes=None, min_ratio=0.0, max_ratio=0.1, random_rate=0.5):
    if random.uniform(0.0, 1.0) < random_rate:
        height, width = image.shape[:2]
        t_pad = int(height * random.uniform(min_ratio, max_ratio))
        b_pad = int(height * random.uniform(min_ratio, max_ratio))
        l_pad = int(width * random.uniform(min_ratio, max_ratio))
        r_pad = int(width * random.uniform(min_ratio, max_ratio))
        image = cv2.copyMakeBorder(image, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        bboxes[:,[0,2]] = bboxes[:,[0,2]] + l_pad
        bboxes[:,[1,3]] = bboxes[:,[1,3]] + t_pad

    return image, bboxes

def random_jpeg_enhance(image, bboxes=None, low_quality=50, high_quality=95, random_rate=0.5):
    if random.uniform(0.0, 1.0) < random_rate:
        quality = random.randint(low_quality, high_quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, image_enc = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(image_enc, 1)
    return image, bboxes

def random_resize(image, bboxes=None, min_scale = 0.5, max_scale = 0.7, flags = 0, random_rate=0.5):
    if random.uniform(0.0, 1.0) < random_rate:
        h, w = image.shape[0], image.shape[1]
        resize_scale = random.uniform(min_scale, max_scale)
        resize_h = int(h * resize_scale)
        resize_w = int(w * resize_scale)
        image = cv2.resize(image, (resize_w, resize_h), flags)
        image = cv2.resize(image, (w, h), flags)
    return image, bboxes

def random_blur(image, bboxes=None, kernel_size=3, random_rate=0.5):
    if random.uniform(0.0, 1.0) < random_rate:
        image = cv2.blur(image, (kernel_size, kernel_size))
    return image, bboxes

def add_bg(img, bg_img, bg_range):
    img = img.astype(np.float32)
    bg_max_val = np.max(bg_img)
    bg_min_val = np.min(bg_img)
    if bg_max_val - bg_min_val < bg_range / 2:
        return img
    bg_img = (bg_img - bg_min_val) / float(bg_max_val - bg_min_val) * bg_range - bg_range/2
    tmp_img = img + bg_img
    width = img.shape[1]

    col_start = random.randint(0, width - 2)
    col_end = random.randint(col_start, width)
    if 1 or 0 == random.randint(0, 1):
        img[:,col_start:col_end,:] = tmp_img[:,col_start:col_end,:]

    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    img = img.astype(np.uint8)
    return img
