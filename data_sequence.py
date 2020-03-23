from keras.utils import Sequence
from voc_label import read_voc_label
import cv2
import random
import numpy as np
class DataSequence(Sequence):

    def __init__(self, labelfile, batch_size, target_height, target_width):
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.path_labels = self.parse_labelfile(labelfile)
        self.cur_index = 0

    def __len__(self):
        count = 0
        for path_label in self.path_labels:
            count += len(path_label[1])
        return count

    def __getitem__(self, idx):
        images = []
        labels = []

        while True:
            if self.cur_index >= len(self.path_labels):
                self.cur_index = 0
            path_label = self.path_labels[self.cur_index]
            image_path, bboxes = path_label
            image = cv2.imread(image_path)
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, classname = bbox
                label_image = self.crop(image, (xmin, ymin, xmax, ymax))
                images.append(label_image)
                labels.append(bbox)
    
                if len(images) == self.batch_size:
                    print(len(images), self.batch_size)
                    print(labels)
                    return np.array(images), labels
            self.cur_index += 1
        return None, None
    def parse_labelfile(self, labelfile):
        lines = self.read_lines(labelfile)
        path_labels = []
        for line in lines:
            image_path = line.split(' ')[0]
            labels = []
            uniq = True
            label_strs = line.split(' ')[1:]
            for index, label_str in enumerate(label_strs):
                xmin, ymin, xmax, ymax, label = label_str.split(',')
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                if index > 0 and label == labels[index-1][4]:
                    uniq = False
                    break
                labels.append((xmin, ymin, xmax, ymax, label))
            if uniq:
                path_labels.append((image_path, labels))
        return path_labels

    def read_lines(self, filepath):
        lines = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
        return lines
    
    def crop(self, image, box_label, thresh=0.4, margin=0.3):
        height, width = image.shape[:2]
        xmin, ymin, xmax, ymax = [int(item) for item in box_label[:4]]
        box_width = xmax - xmin
        box_height = ymax - ymin

        if xmin > width * thresh:
            start = max(0, int(xmin - box_width))
            end = int(xmin - margin * box_width)
            xmin = random.randint(start, end)
        else:
            xmin = random.randint(0, xmin)

        if ymin > height * thresh:
            start = max(0, int(ymin - box_height))
            end = int(ymin - margin * box_height)
            ymin = random.randint(start, end)
        else:
            ymin = random.randint(0, ymin)
        
        if xmax < width - thresh * box_width:
            start = int(xmax + margin * box_width)
            end = min(width, xmax + box_width)
            xmax = random.randint(start, end)
        else:
            xmax = random.randint(xmax, width)
        
        if ymax < height - thresh * box_height:
            start = int(ymax + margin * box_height)
            end = min(height, ymax + box_height)
            ymax = random.randint(start, end)
        else:
            ymax = random.randint(ymax, height)

        crop_image = image[ymin:ymax, xmin:xmax].copy()
        crop_image = cv2.resize(crop_image, (self.target_height, self.target_width))
        return crop_image

if __name__ == '__main__':
    data_sequence = DataSequence('label_lines.txt', 2, 227, 227)
    image_path = 'images/_MG_0001.JPG'
    voc_label_path = 'images/_MG_0001.xml'
    voc_label = read_voc_label(voc_label_path)
    image = cv2.imread(image_path)
    data_sequence.crop(image, voc_label[0])
    images, labels = data_sequence.__getitem__(0)
    print(images.shape)
    print(labels)

