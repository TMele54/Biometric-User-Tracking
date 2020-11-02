import sys
import os
import dlib
import cv2
import glob
import argparse
import pickle
from math import sqrt

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

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Instance of Tracker  "tracking_goturn/goturn.caffemodel/goturn.caffemodel"
tracker = cv2.TrackerGOTURN_create()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

bbox = None

while True:

    # Capture frame-by-frame
    ok, frame = cap.read()

    # Uncomment the line below to select a different bounding box
    if bbox == None:
        bbox = test_frame(frame)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, "GOTURN Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
