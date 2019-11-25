#coding=utf-8
import codecs
import cv2
import numpy as np
from random import uniform, randint
import tensorflow as tf
from resnet_v1 import resnet_v1_50
from data import generator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

slim = tf.contrib.slim


def read_lines(filepath):
    lines = []
    with codecs.open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
    return lines

def to_one_hot(label, num_classes):
    one_hot_label = np.zeros(num_classes, dtype=np.float)
    one_hot_label[label] = 1.0
    return one_hot_label

def rand_crop(image):
    height, width = image.shape[0], image.shape[1]
    h_crop = int(uniform(0.8, 1.0) * height)
    w_crop = int(uniform(0.8, 1.0) * width)
    h_start = randint(0, height - h_crop)
    w_start = randint(0, width - w_crop)
    image = image[h_start:h_start+h_crop, w_start:w_start+w_crop]
    return image

def generator(filepath, height, width, num_classes, batch_size):
    lines = read_lines(filepath)

    index = 0
    while True:
        images = []
        labels = []

        for i in range(batch_size):
            if index >= len(lines):
                index = 0

            line = lines[index]
            line = line.split(u' ')
            image_path = line[0]
            image = cv2.imread(image_path)
            if image is None:
                print(line[0])
                print(line[1])
            image = rand_crop(image)
            image = cv2.resize(image, (width, height))
            image = image.astype(np.float32)
            image -= 128
            image /= 255.0
            label = to_one_hot(int(line[1]), num_classes)
            
            images.append(image)
            labels.append(label)
            index += 1

        yield np.array(images), np.array(labels)

if __name__ == '__main__':
    num_classes = 70
    is_training = True
    height = 256
    width = 256
    batch_size = 16
    inputs = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    targets = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    learning_rate = 0.001
    total_steps = 100000


    
    end_points = my_net(inputs, num_classes, is_training=is_training)
    #predictions = end_points['predictions']
    predictions = end_points['fc2']
    #predictions = tf.squeeze(predictions, axis=[1,2])
    correct_prediction=tf.equal(tf.argmax(predictions,axis=1), tf.argmax(targets,axis=1))#shape of correct_prediction is [N]
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets,  logits=predictions)) 
    tf.summary.scalar('loss', loss) 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_steps = tf.Variable(0, trainable=False)
    train_step = optimizer.minimize(loss, global_step=global_steps)

    filepath = 'train.txt'
    gen = generator(filepath, height, width, num_classes, batch_size)

    saver = tf.train.Saver()
    init=tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        for step in range(total_steps):
            images_, labels_ = next(gen)
            sess.run(train_step, feed_dict={inputs:images_, targets:labels_})
            loss_, accuracy_ = sess.run([loss, accuracy], feed_dict={inputs:images_, targets:labels_})
            print('loss: %f, accuracy: %f'% (loss_, accuracy_))
            if 0 == step % 5000:
                saver.save(sess, 'ckpt/', global_step=step)

