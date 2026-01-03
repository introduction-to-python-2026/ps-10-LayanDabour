from image_utils import load_image
from image_utils import edge_detection
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

path = "/content/IMG_0885.jpg"
image = load_image(path)
clean_image=median(image, ball(3))
edgmag=edge_detection(clean_image)
plt.hist(edgmag.ravel(), bins=100) 
plt.show()

thr = np.percentile(edgmag, 95)
plt.imshow(edgmag>thr, cmap='gray')
plt.axis('off')
plt.show()
print(edgmag.shape)

edge_image = Image.fromarray(edgmag>thr)
edge_image.save('my_edges.png')
