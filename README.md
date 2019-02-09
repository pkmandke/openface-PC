# Openface(Facenet CNN) real-time implementation

## Reference

The pre-trained model & it's associated pre-processing is credited to [the openface implementation](https://github.com/cmusatyalab/openface). Further, the Facenet CNN model architecture has been inspired by [this](https://arxiv.org/abs/1503.03832) paper by Schroff et al.

## What is in this repo?

This repository provides helper scripts that implement the openface CNN for real-time face recongition on a PC using an external webcam(Logitech). The pre-trained CNN architecture used is the nn4.small2.v1 by [openface](https://github.com/cmusatyalab/openface).
The faces have been cropped but not warped using an affine transform as is done in the original model by [openface](https://github.com/cmusatyalab/openface).


Running main.py in /scripts starts the API for adding a new face and/or detecting a pre-trained face. The path/s to the pre-trained CNN and the face detection model by [dlib](http://dlib.net/) need to be added manually in the main.py file as the right varibales.


# Author

[Prathamesh Mandke](https://www.linkedin.com/in/prathamesh-mandke-866866168/)
