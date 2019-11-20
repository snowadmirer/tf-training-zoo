#coding=utf-8
import tensorflow as tf
import numpy as np
import cv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def add_to_record(image_bytes, label_bytes, record_writer):
    example=tf.train.Example(features=tf.train.Features(feature={
    'image_bytes':_bytes_feature(image_bytes),
    'label_bytes':_bytes_feature(label_bytes)}))
    record_writer.write(example.SerializeToString())

def _parse_function(example_proto):
    feature_description = {
        'image_bytes': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label_bytes': tf.io.FixedLenFeature([], tf.string, default_value=''),}
    return tf.io.parse_single_example(example_proto, feature_description)

if __name__ == '__main__':
    tf.contrib.eager.enable_eager_execution()
    filename = 'test.tfrecord'
    writer=tf.io.TFRecordWriter(filename)
    image = np.zeros((512,512,3), dtype=np.uint8)
    label = np.zeros(512, dtype=np.float32)

    image_bytes = image.tobytes()
    label_bytes = label.tobytes()
    add_to_record(image_bytes, label_bytes, writer)
    add_to_record(image_bytes, label_bytes, writer)
    add_to_record(image_bytes, label_bytes, writer)
    add_to_record(image_bytes, label_bytes, writer)
    writer.close()

    filenames = ['record/0.tfrecord']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print(raw_dataset)
    for raw_record in raw_dataset.take(4):
        print(type(raw_record))
    parsed_dataset = raw_dataset.map(_parse_function)
    print(parsed_dataset)
    for parsed_record in parsed_dataset.take(1):
        image_bytes = parsed_record['image_bytes']
        label_bytes = parsed_record['label_bytes']
        image_decode = np.frombuffer(image_bytes.numpy(), dtype=np.uint8)
        image_decode = np.frombuffer(image_bytes.numpy(), dtype=np.uint8)
        image_decode = image_decode.reshape((1024,1024,3))
        label_decode = np.frombuffer(label_bytes.numpy(), dtype=np.float32)
        print((image == image_decode).all())
        print((label == label_decode).all())
