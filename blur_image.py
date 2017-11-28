# Image Blur implemented using Tensorflow

import numpy as np

import tensorflow as tf
from PIL import Image
from smoother import Smoother

# Basic model parameters.
tf.app.flags.DEFINE_string('image_path', './Colosseum_in_Rome,_Italy_-_April_2007.jpg',
                           """Path to the image to blur.""")
FLAGS = tf.app.flags.FLAGS

# Basic Constants
SIGMA = 2.0
FILTER_SIZE = 13

def smooth():

    Image_Placeholder = tf.placeholder( tf.float32, shape = [1, None, None, 3])
    smoother = Smoother({'data':Image_Placeholder}, FILTER_SIZE, SIGMA)
    smoothed_image = smoother.get_output()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        image = Image.open(FLAGS.image_path)
        image = np.array(image, dtype = np.float32)
        image = image.reshape((1, image.shape[0], image.shape[1], 3))
        smoothed = sess.run(smoothed_image,
                             feed_dict = {Image_Placeholder: image})
        smoothed = smoothed / np.max(smoothed)
        out_image = np.squeeze(smoothed)

        out_image = Image.fromarray(np.squeeze(np.uint8(out_image * 255)))
        out_image.show()


def main(argv=None):
  smooth()

if __name__ == '__main__':
  tf.app.run()
