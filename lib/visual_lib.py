import random
import pandas as pd
import numpy as np
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2

def visualize_with_rect(image, rect):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for r in rect:
      image = cv2.rectangle(image, (r[0],r[1]), (r[2], r[3]), (255, 0, 0), 2) 
    plt.figure(figsize=(10, 20))
    plt.imshow(image)

def visualize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 20))
    plt.imshow(image)