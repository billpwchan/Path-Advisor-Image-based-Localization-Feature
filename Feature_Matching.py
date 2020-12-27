import argparse
import glob
import os
import timeit

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Sample Calling Script => python Feature_Matching.py -qd ./query/query_image.jpg -td ./dataset/training_set_archived
parser = argparse.ArgumentParser(
    description='HKUST Path Advisor SIFT Feature Matching Scripts')

parser.add_argument('-qd', '--query_dir', required=True,
                    help="<Required> query image file path", type=str)
parser.add_argument('-td', '--train_dir', required=True,
                    help="<Required> trainin data file path", type=str)
parser.add_argument('-rl', '--room_list', nargs='+',
                    help='<Required> room list (5 locations should be provided)', required=True)
parser.add_argument('-m', '--min_match', required=False, default=10,
                    help="<Optional> minimum matched feature requirement")
parser.add_argument('-d', '--draw', required=False,
                    default=False, help="<Optional> draw matching diagram", action="store_const")

args = parser.parse_args()

# Initialize Variables
MIN_MATCH_COUNT = int(args.min_match)
if os.path.exists(args.train_dir):
    TRAINING_IMAGE_PATH = args.train_dir
else:
    raise IOError("Invalid Training Image Path Provided")
if os.path.exists(args.query_dir):
    QUERY_IMAGE_PATH = args.query_dir
else:
    raise IOError("Invalid Query Image Path Provided")

# ROOMLIST = [name for name in os.listdir('dataset/training_set_archived/')]
ROOMLIST = args.room_list

# reading images in grayscale format
query_image = cv.imread('dataset/training_set_archived/LTB/LTB-d.png', 0)


def calculateFeatureMatching(training_image, query_image):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(training_image, None)
    kp2, des2 = sift.detectAndCompute(query_image, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    exactMatches = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            exactMatches[i] = [1, 0]
    if (args.draw):
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=exactMatches,
                           flags=0)
        img3 = cv.drawMatchesKnn(training_image, kp1, query_image, kp2,
                                 matches, None, **draw_params)
        resultFig = plt.figure(figsize=(5, 5))
        ax3 = resultFig.add_subplot(111)
        ax3.imshow(img3, 'gray'), plt.show()
        resultFig.savefig('example.png', dpi=1000)
    return len(exactMatches)


maxMatch = []
start = timeit.default_timer()

for roomname in ROOMLIST:
    filenames = glob.glob(TRAINING_IMAGE_PATH + roomname + '/*-[abcdm].*')
    for training_image in filenames:
        exactMatches = calculateFeatureMatching(
            cv.imread(training_image, 0), query_image)
        maxMatch.append((str(training_image), int(exactMatches)))

stop = timeit.default_timer()
print('Time: {}'.format(stop - start))

maxMatch.sort(key=lambda x: x[1], reverse=True)
print(maxMatch)
