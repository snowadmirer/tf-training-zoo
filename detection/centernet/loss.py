#coding=utf-8
from lxml import etree
import tensorflow as tf
import numpy as np
from glob import glob
import cv2

from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian



def read_xml(xml_file):
    with open(xml_file) as f:
        return etree.fromstring(f.read())

def resize(image, box_labels, resize_w, resize_h):
    h, w = image.shape[0], image.shape[1]
    w_scale = float(resize_w) / w
    h_scale = float(resize_h) / h
    image = cv2.resize(image, (resize_w, resize_h))
    tmp_box_labels = []
    for box_label in box_labels:
        xmin, ymin, xmax, ymax = box_label[:4]
        xmin, xmax = w_scale * xmin, w_scale * xmax
        ymin, ymax = h_scale * ymin, h_scale * ymax
        tmp_box_labels.append([xmin, ymin, xmax, ymax, box_label[4]])

    return image, tmp_box_labels

def expand_feature(center_x, center_y, box_width, box_height, feature_width, feature_height):
    hm = np.zeros((feature_height, feature_width), dtype=np.float32)
    center_x_int = int(center_x)
    center_y_int = int(center_y)
    radius = gaussian_radius((box_width, box_height))
    radius = int(radius)
    ct = np.array([center_x, center_y], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    draw_umich_gaussian(hm, ct_int, radius)
    return hm

def make_feature_targets(image, box_labels, num_classes, stride):
    height, width = image.shape[0], image.shape[1]
    feature_height = int(height // stride)
    feature_width = int(width // stride)
    features = [[np.zeros((feature_height, feature_width), dtype=np.float32)] for i in range(num_classes)]

    for box_label in box_labels:
        xmin, ymin, xmax, ymax, label = box_label
        center_x = (xmin + xmax) / stride / 2.0
        center_y = (ymin + ymax) / stride / 2.0
        box_width = (xmax - xmin) / stride
        box_height = (ymax - ymin) / stride
        center_x_int = int(center_x)
        center_y_int = int(center_y)
        feature_expansion = expand_feature(center_x, center_y, box_width, box_height, feature_width, feature_height)
        features[label].append(feature_expansion)
    
    for i in range(num_classes):
        if len(features[i]) > 1:
            features[i] = np.max(np.array(features[i]), axis=0)
        else:
            features[i] = features[i][0]
    features = np.array(features)
    return features

def make_coordinate_targets(image, box_labels, num_classes, stride):
    height, width = image.shape[0], image.shape[1]
    feature_height = int(height // stride)
    feature_width = int(width // stride)

    coordinate_targets = np.zeros((num_classes, feature_height, feature_width), dtype=np.float32)
    for box_label in box_labels:
        xmin, ymin, xmax, ymax, label = box_label
        center_x = (xmin + xmax) / stride / 2.0
        center_y = (ymin + ymax) / stride / 2.0
        box_width = (xmax - xmin) / stride
        box_height = (ymax - ymin) / stride
        center_x_int = int(center_x)
        center_y_int = int(center_y)
        coordinate_targets[label, center_y_int, center_x_int] = 1.0
    return coordinate_targets

def make_size_targets(image, box_labels, num_classes, stride):
    height, width = image.shape[0], image.shape[1]
    feature_height = int(height // stride)
    feature_width = int(width // stride)
    size_targets = np.zeros((feature_height, feature_width, 2), dtype=np.float32)
    
    for box_label in box_labels:
        xmin, ymin, xmax, ymax, label = box_label
        center_x = (xmin + xmax) / stride / 2.0
        center_y = (ymin + ymax) / stride / 2.0
        box_width = (xmax - xmin) / stride
        box_height = (ymax - ymin) / stride
        center_x_int = int(center_x)
        center_y_int = int(center_y)
        size_targets[center_y_int, center_x_int, 0] = box_height
        size_targets[center_y_int, center_x_int, 1] = box_width
    return size_targets

def make_offset_targets(image, box_labels, num_classes, stride):
    height, width = image.shape[0], image.shape[1]
    feature_height = int(height // stride)
    feature_width = int(width // stride)
    offset_targets = np.zeros((feature_height, feature_width, 2), dtype=np.float32)
    
    for box_label in box_labels:
        xmin, ymin, xmax, ymax, label = box_label
        center_x = (xmin + xmax) / stride / 2.0
        center_y = (ymin + ymax) / stride / 2.0
        box_width = (xmax - xmin) / stride
        box_height = (ymax - ymin) / stride
        center_x_int = int(center_x)
        center_y_int = int(center_y)
        offset_targets[center_y_int, center_x_int, 0] = center_y - center_y_int
        offset_targets[center_y_int, center_x_int, 1] = center_x - center_x_int
    return offset_targets

if __name__ == '__main__':
    voc_img_dir = 'F:/voc/VOCdevkit/VOC2007/JPEGImages/'
    image_paths = glob(voc_img_dir + '*.jpg')
    image_paths.sort()
    num_classes = 10
    stride = 4.0
    for image_path in image_paths:
        image = cv2.imread(image_path)
        label_path = image_path.replace('.jpg', '.xml').replace('JPEGImages', 'Annotations')
        print(label_path)
        xml_label = read_xml(label_path)
        print(xml_label)
        box_labels = []
        for label_index, label in enumerate(xml_label.findall('object')):
            label_name = label.find('name').text
            bndbox = label.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            print(label_name)
            print(xmin, ymin, xmax, ymax)
          
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            box_labels.append([xmin, ymin, xmax, ymax, label_index])
        image, box_labels = resize(image, box_labels, 512, 512)
        num_labels = len(box_labels)
        feature_targets = make_feature_targets(image, box_labels, num_classes, stride)
        coordinate_targets = make_coordinate_targets(image, box_labels, num_classes, stride)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i in range(num_classes):
            hm_plot = (feature_targets[i,...] * 255).astype(np.uint8)
            image += hm_plot
            cv2.imwrite('features/{}.png'.format(i), hm_plot)
            co_plot = (coordinate_targets[i,...] * 255).astype(np.uint8)
            cv2.imwrite('coordinates/{}.png'.format(i), co_plot)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        break
    pass

