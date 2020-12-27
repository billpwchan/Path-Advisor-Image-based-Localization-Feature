import pickle
import argparse
import glob
import heapq
import multiprocessing
import operator
import os
import sys
import timeit

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from joblib import Parallel, delayed
from keras.preprocessing.image import ImageDataGenerator

# Sample Calling Script =>  python Location_Estimation.py -qd ./Query/Test/1504-b.png -td ./dataset/training_set/ -lm ./model/model_newest.h5 -m 10 -mode h -kd ./keypoints_database.p
parser = argparse.ArgumentParser(
    description='HKUST Path Advisor CNN Location Estimation Script')

parser.add_argument('-qd', '--query_dir', required=True,
                    help="<Required> query image file path", type=str)
parser.add_argument('-td', '--train_dir', required=True,
                    help="<Required> trainin data file path", type=str)
parser.add_argument('-lm', '--load_model', required=True,
                    help="<Required> model file path", type=str)
parser.add_argument('-kd', '--keypoint_database', required=True,
                    help="<Required> Keypoint database file path", type=str)
parser.add_argument('-m', '--min_match', required=False, default=10,
                    help="<Optional> minimum matched feature requirement")
parser.add_argument('-mode', '--mode', required=True,
                    help="<Required> Use CNN (c) or FLANN-based Feature Matching (f) for Location Estimation", choices=['c', 'f', 'h'])
args = parser.parse_args()

print("<INFO> Argument Received.")
print(sys.argv[1:])

if os.path.exists(args.query_dir):
    query_dir = args.query_dir
else:
    raise IOError("Invalid Query Path Provided")
if os.path.exists(args.load_model):
    model_dir = args.load_model
else:
    raise IOError("Invalid Model Path Provided")
if os.path.exists(args.train_dir):
    train_data_dir = args.train_dir
else:
    raise IOError("Invalid Training Path Provided")
if os.path.exists(args.keypoint_database):
    keypoint_database_dir = args.keypoint_database
else:
    raise IOError("Invalid Keypoint Database Path Provided")
MIN_MATCH_COUNT = int(args.min_match)


print("<INFO> Creating Room Name <==> Model Class ID Mapping")
# A quick script to get the room name => model class id mapping
class_names = glob.glob(train_data_dir + "*")
class_names = sorted([os.path.basename(os.path.normpath(name))
                      for name in class_names])  # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
name_id_map = {v: k for k, v in name_id_map.items()}

print("<INFO> Reading Query Image")
query_image = image.load_img(query_dir, target_size=(299, 299))
query_image = np.expand_dims(query_image, axis=0)

predicted_labels = []
maxMatch = []


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    image_class = None
    for point in array:
        temp_feature = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                   _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        image_class = point[7]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors), image_class


def calculateFeatureMatching(kp1, desc1, kp2, desc2, img_class):
    # Initiate SIFT detector

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    # store all the good matches as per Lowe's ratio test.
    exactMatches = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            exactMatches[i] = [1, 0]
    return int(exactMatches.count([1, 0]))


if args.mode == 'c' or args.mode == 'h':
    print("<INFO> Using CNN Image Recognition for Location Estimation")
    import tensorflow as tf
    from keras.models import Model, load_model

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # load model
    model = load_model(model_dir)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory('Query/',
                                                      target_size=(299, 299),
                                                      batch_size=1,
                                                      shuffle=False,
                                                      class_mode='categorical')
    test_generator.reset()
    y_prob = model.predict_generator(
        test_generator, verbose=0, steps=test_generator.samples/1, workers=1)

    # Always return the top 1 first, decending order (Top 5 Predictions)
    y_class_top5 = list(
        zip(*heapq.nlargest(5, enumerate(y_prob[0]), key=operator.itemgetter(1))))[0]

    # To get the Top 1 result from the previous array ([24]) => Receive the maximum value for each row
    y_class_top1 = y_prob.argmax(axis=-1)

    predicted_label = name_id_map[int(y_class_top1)]
    predicted_labels = [(name_id_map[int(index)], round(
        y_prob[0][index], 2)) for index in y_class_top5]
    print("<Result> Predicted Rooms from CNN:")
    print(predicted_labels)

# Following Scripts for calculating FLANN-based Feature Matching


if args.mode == 'f' or args.mode == 'h':
    print("<INFO> Using FLANN-based Feature Matching for Location Estimation")
    # reading images in grayscale format
    keypoints_database = pickle.load(open(keypoint_database_dir, "rb"))
    query_image = cv.imread(query_dir, 0)
    sift = cv.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(query_image, None)
    maxMatch = {}
    start = timeit.default_timer()
    num_cores = multiprocessing.cpu_count()
    # predicted_labels = [[item[1], ] for item in name_id_map.items()]

    for keypoint in keypoints_database:
        kp2, desc2, img_class = unpickle_keypoints(keypoint)
        maxMatch[img_class] = max(maxMatch.get(
            img_class, 0), calculateFeatureMatching(kp1, desc1, kp2, desc2, img_class))

    stop = timeit.default_timer()
    print('Time: {}'.format(stop - start))

    maxMatch = [(k, v) for k, v in maxMatch.items()]
    maxMatch.sort(key=lambda x: x[1], reverse=True)
    maxMatch = maxMatch[:5]
    summation = sum([item[1] for item in maxMatch])
    maxMatch = [(item[0], item[1]/summation) for item in maxMatch]
    print("<Result> Predicted Rooms from FLANN-based Matching:")
    print(maxMatch)

if args.mode == 'h':
    hybrid_pred = {}
    for item in predicted_labels:
        hybrid_pred[item[0]] = hybrid_pred.get(item[0], 0) + item[1]
    for item in maxMatch:
        hybrid_pred[item[0]] = hybrid_pred.get(item[0], 0) + item[1]
    hybrid_pred = [(k, v/2) for k, v in hybrid_pred.items()]
    hybrid_pred.sort(key=lambda x: x[1], reverse=True)
    hybrid_pred = hybrid_pred[:5]
    print("<Result> Predicted Rooms from Hybrid-Mode:")
    print(hybrid_pred)
