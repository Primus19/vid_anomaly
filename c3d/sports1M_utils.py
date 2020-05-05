# -*- coding: utf-8 -*-
"""Preprocessing tools for C3D input videos

"""

import numpy as np
import cv2
from keras.utils.data_utils import get_file
import pdb

C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'
SPORTS1M_CLASSES_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_classes.txt'

def preprocess_input(video):
    """Resize and subtract mean from video input

    Keyword arguments:
    video -- video frames to preprocess. Expected shape
        (frames, rows, columns, channels). If the input has more than 16 frames
        then only 16 evenly samples frames will be selected to process.

    Returns:
    A numpy array.

    """
    print('video shape:',video.shape)
    intervals = np.ceil(np.linspace(0, video.shape[0]-1, 16)).astype(int)
    frames = video[intervals]


    height=128 #rows
    width=171 #columns
    #channels = img.shape[2] #channels usually 3.
    dim=(width,height)  #cv2 use (columns,row) as default format
    # Reshape to 128x171
    #skvideo use: (frame,row,columns,channels) as default format
    reshape_frames = np.zeros((frames.shape[0], height, width, frames.shape[3]))
    print('reshape frames:',reshape_frames.shape)

    for i, img in enumerate(frames):
         img = cv2.resize(img , dim) #the resize is dim=height,width
         #print('dimensions',dim)
         #print('cv2 shape img:',img.shape)
         reshape_frames[i,:,:,:] = img

    mean_path = get_file('c3d_mean.npy',
                         C3D_MEAN_PATH,
                         cache_subdir='models',
                         md5_hash='08a07d9761e76097985124d9e8b2fe34')

    #pdb.set_trace()
    # Subtract mean
    mean = np.load(mean_path)
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:,8:120,30:142,:]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)

    return reshape_frames

def decode_predictions(preds):
    """Returns class label and confidence of top predicted answer

    Keyword arguments:
    preds -- numpy array of class probability

    Returns:
    A list of tuples.

    """
    class_pred = []
    for x in range(preds.shape[0]):
        class_pred.append(np.argmax(preds[x]))

    labels_path = get_file('sports1M_classes.txt',
                           SPORTS1M_CLASSES_PATH,
                           cache_subdir='models',
                           md5_hash='c102dd9508f3aa8e360494a8a0468ad9')

    with open(labels_path,  encoding="utf8") as f:
        labels = [lines.strip() for lines in f]

    decoded = [(labels[x],preds[i,x]) for i,x in enumerate(class_pred)]

    return decoded
