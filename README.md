# Biometric User Tracking
> This repo uses several different Object Tracking Patterns to track a person's motion in real time with very strong reliability. These Algorithms are initiated using Facial Recognition Landmark Detection and Eucluian Distance via an input image 'key' to ensure the correct user is being tracked. OpenCV Trackers Included: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'

I built this package to lay the ground work for building a high performance UAV from scratch with advanced Computer Vision applications on board. All that is needed to use this package in a single image of the indended trackee's face. The code will analyze the user's input image and define a 128 Dimension Vector which is saved to disk. When you execute the multitrack algorithm, an initial frame is captured and all users in the image are scanned. Minimum Eucludian Distance is used to identify and track the intended user. Once the individual is identified, the facial recognition and landmark algorithms are scrapped and the algorithm runs purely with motion tracking. This allows trackers like 'GOTURN' to track the user without requiring the person's face to be in the frame. For example, you may want to follow the user while they are riding a bike.

![](header.png)

## Installation

Windows/ Linux (Beagle Bone, Arduiono):

```sh
git clone https://github.com/TMele54/BiometricUserTracking.git
```

## Usage example

Execute the following:

Replace tony.jpg in the encoding_images folder with a new .jpg image of a user's face. 

```sh
git add encoding_images/

python3 start.py --images=encoding_images

cd multitrack/

python3 multitrack.py
```
##

If you would like to track something like a backetball, Execute the following:

```sh
cd multitrack/

python3 multiChipTrack.py
```

Use the GUI to define a region of interest around the Basketball

## Development setup

This requires dlib and OpenCV.

OpenCV (Windows & Linux): ```sh pip install opencv-python opencv-contrib-python ```

Dlib (Windows): https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f
dlib (Ubuntu): https://www.learnopencv.com/install-dlib-on-ubuntu/
 
## Meta

Anthony Mele – tony@datasciencenewyork.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/TMele54/BiometricUserTracking] (https://github.com/TMele54/)

 
