#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:14:13 2018

@author: joni33
"""

import re
import numpy
import torch

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    
    
     
    
    RPM_FL, RPM_FR, RPM_RL, RPM_RR, Yaw = read_velocity(header)
    
     
    
    return [RPM_FL, RPM_FR, RPM_RL, RPM_RR, Yaw], resize(numpy.frombuffer(buffer,
                            dtype='int' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width))), 80, 60)
    
    

def resize(image, i_width, i_height):
    import scipy.misc
    return scipy.misc.imresize(image, (i_height, i_width))


def read_velocity(header):
    
    words = header.split()
    
    for i, w in enumerate(words):
        if w == "#WheelRPM_FR:":
            FR = float(words[i+1])
        elif w == "#WheelRPM_FL:":
            FL = float(words[i+1])
        elif w == "#WheelRPM_RL:":
            RL = float(words[i+1])
        elif w ==  "#WheelRPM_RR:":
            RR = float(words[i+1])
        elif w == "#YawRate=":
            Yaw = float(words[i+1])
    return FL, FR, RL, RR, Yaw
            
    
if __name__ == "__main__":
    from matplotlib import pyplot
    
    action, image = read_pgm("/home/joni33/prednet_driving/Reinhard_data/ConstructionSite-left/image0001_c0.pgm", byteorder='<')
    action = torch.Tensor(action)
    img1_corr = (image) /65535.
    #(image / 65535.) # **(1/2.2)
    # print img1_corr
    
    pyplot.imshow(img1_corr, pyplot.cm.gray)
    pyplot.show()