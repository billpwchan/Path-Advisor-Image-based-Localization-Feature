import argparse
import heapq
import os
import sys
from glob import glob
import operator

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from keras import backend as k
from keras.applications import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model = load_model("./model/model_newest.h5")

from glob import glob
class_names = glob("./dataset/training_set/*") # Reads all the folders in which images are present
class_names = sorted([os.path.basename(os.path.normpath(name)) for name in class_names]) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
name_id_map =  {v: k for k, v in name_id_map.items()}

# The below script is used for calculating accuracy of the class in prediction. 

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory('dataset/test_set/',
                                                              target_size=(299, 299),
                                                              batch_size=1,
                                                              shuffle=False,
                                                              class_mode='categorical')
test_generator.reset()
Y_pred = model.predict_generator(test_generator, verbose=1, steps=test_generator.samples/1, workers=1)
classes = test_generator.classes[test_generator.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
print('Confusion Matrix')
output_confusion_matrix = confusion_matrix(classes, y_pred)
output_classification_report = classification_report(test_generator.classes, y_pred)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(output_confusion_matrix, index = [name_id_map[i] for i in range(116)],
                  columns = [name_id_map[i] for i in range(116)])
# plt.figure(figsize = (1920,1080))
sn.set(font_scale=1.4)
fig = plt.figure(figsize=(100, 70))
sn.heatmap(df_cm, annot=True, annot_kws={"size":16})
fig.savefig('myimage.svg', format='svg', dpi=300)