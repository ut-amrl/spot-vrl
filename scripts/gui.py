import cv2
import numpy as np
import cluster_jackal
import 25_train_jackal

# OpenCV GUI for the project
def display_image(image, title):
    """Displays an image in a window
    Arguments:
        image: the image to display
        title: the title of the window
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_images(images, titles):
    """Displays multiple images in a window
    Arguments:
        images: the images to display
        titles: the titles of the images
    """
    for i in range(len(images)):
        display_image(images[i], titles[i])

def get_samples(data):


    clusters , elbow = cluster_jackal.cluster(data)