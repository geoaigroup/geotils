import sys
from geotils.data_processing.pre_processing import LargeTiffLoader
import numpy as np
import matplotlib.pyplot as plt
loader = LargeTiffLoader(
    r"\assets",
    image_suffix=".tiff",
)
loader.load_index(
    r"\output",
    1000,
    1000,
    1024,
    1024,
)
"""requires experimenting """
imgs = loader.pre_load(
   r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotorch\archive\tiff\test_labels",
   fragment_size=512,
   mask_suffix=".tif",
)

plt.imshow(imgs)
plt.axis("off")  # Hide the axis
plt.title("Image")
plt.show()
