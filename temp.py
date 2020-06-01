import tensorflow as tf 
import tensorflow_datasets as tfds 
import numpy as np
# dataset, info = tfds.load("mnist", as_supervised= True, with_info=True) 

# train_data = dataset['train'] 
# for image, label in train_data.take(1):
#     break 
# print(image)
# t = image.numpy().shape()
# print(t)
dataset = tf.data.Dataset.range(8)
# print(tf.shape(dataset))
dataset = dataset.batch(3)

# t = dataset.as_numpy_iterator()
# print(list(t))
# # print(tf.shape(dataset).as_list())
# for arr in dataset.take(1):
#     break
# print(arr)
# tmp = dataset.map(lambda d:d)
# print(tmp)
print(tf.shape(dataset))