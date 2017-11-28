# Image Segmentation and Blur using Open CV and Tensorflow

## Dependencies

1. Numpy
2. Scipy
3. Tensorflow
4. Pillow
5. Open CV
6. Matplotlib

## Class image-segment and blur-open cv

The file `image-segment and blur-open cv.py` contains the class to implement the segmentation and blurring on the feeded input using open cv. 

## Class segmentation with scikit

The file `segmentation with scikit.py` contains the class to implement the segmenation on the feeded input using scikit learn. 

## Class Smoother

The file `smoother.py` contains the main class to implement the blurring on the feeded input using tensorflow. 
This input can be either an image (as shown in the example) or one of the layers of a Neural Network.
The main class to import is `Smoother`. The class is a forked version of [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)

## Run

To run the example:

Segmentation and Blur using Open CV
`python image-segment and blur-open cv.py`

Segementation with Scikit Learn
`python segmentation with scikit.py`

Blurring using Tensorflow
`python blur_image.py`
