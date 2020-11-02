from __future__ import print_function
from random import randint
from math import sqrt
import argparse
import pickle
import dlib
import glob
import sys
import cv2
import os

def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmarks', '--L', type=str, default="../dlib_models/landmarks.dat", help='face landmarks')
    parser.add_argument('--resnet', '--R', type=str, default="../dlib_models/resnet.dat", help='residual network')
    parser.add_argument('--images', '--I', type=str, default="../test_images", help='Confounding image files')
    parser.add_argument('--username', '--U', type=str, default="tony", help='User file name')
    args = parser.parse_args()

    L = args.landmarks
    R = args.resnet
    I = args.images
    U = args.username

    return L,R,I,U
def get_encoding(fname):
    with open('../encoding_data/'+fname+".p", 'rb') as f:
        user_attrs = pickle.load(f)
    return user_attrs
def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))
def test_frame(f):
    dets = detector(f, 1)
    face_descriptors = []

    # Now process each face we found.
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(f, d)

        face_chip = dlib.get_face_chip(f, shape)
        face_descriptor = facerec.compute_face_descriptor(face_chip)
        face_descriptors.append([face_descriptor, d])

    scores = []
    for face in face_descriptors:
        scores.append(euclidean_dist(face[0], user_encoding[0]["face_descriptor"]))

    index_min = min(range(len(scores)), key=scores.__getitem__)

    return (face_descriptors[index_min][1].left(),
                face_descriptors[index_min][1].top(),
                face_descriptors[index_min][1].right(),
                face_descriptors[index_min][1].bottom())

# Input Args
predictor_path, face_rec_model_path, test_images, username = command_line_options()
print("Models Loaded")

# Retrieve user encoding
user_encoding = get_encoding(username)

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

# Create a video capture object to read videos
cap = cv2.VideoCapture(0)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  if (k == 113):  # q is pressed
    break

print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
trackerType = "CSRT"

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
