import sys
import os
import dlib
import glob
import argparse
import pickle

def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmarks', '--L', type=str, default="dlib_models/landmarks.dat", help='face landmarks')
    parser.add_argument('--resnet', '--R', type=str, default="dlib_models/resnet.dat", help='residual network')
    parser.add_argument('--images', '--I', type=str, default="examples/faces", help='example images')
    parser.add_argument('--username', '--U', type=str, default="tony", help='User file name')
    args = parser.parse_args()

    L = args.landmarks
    R = args.resnet
    I = args.images
    U = args.username

    return L,R,I,U
def save_encoding(fname, v1n, v1v, v2n, v2v):
    with open("encoding_data/"+fname+".p", 'wb') as f:
        pickle.dump([{v1n: v1v, v2n: v2v}], f)
    return True
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

predictor_path, face_rec_model_path, faces_folder_path, username = command_line_options()
print("Models Loaded")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# Now process all the images
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        bb_dim = rect_to_bb(d)

        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print("face_descriptor:", face_descriptor)
        print("Computing descriptor on aligned image ..")

        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img, shape)

        # Now we simply pass this chip (aligned image) to the api
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
        print("face_descriptor_from_prealigned_image",face_descriptor_from_prealigned_image)

        # Save Encoding
        save_encoding(username, "face_descriptor", face_descriptor_from_prealigned_image, "bb", bb_dim)

        dlib.hit_enter_to_continue()
