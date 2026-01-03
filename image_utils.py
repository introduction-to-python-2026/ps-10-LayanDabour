from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
  image = Image.open(path)
  image = np.array(image)
  return image

def edge_detection(image):
  if image.ndim == 3:                
        gray = np.mean(image, axis=2)
  else:                               
        gray = image
  gray = gray.astype(np.float32)
  KernelY = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
  KernelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  edge_x=convolve2d(gray, KernelX, mode='same', boundary='fill')
  edge_y=convolve2d(gray, KernelY, mode='same', boundary='fill')
  edgeMAG=(edge_x**2 + edge_y**2)**0.5
  return edgeMAG

