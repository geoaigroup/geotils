import numpy as np
import random


class Visualization(object):
    """enter an image and mask to be able to modify them
    the img and mask will  be themselves modified and a modified version will be returned after every call
    """

    def __init__(self, img, mask):
        self.mask = mask
        self.img = img

    """This method converts the mask to an RGB representation.
        If the mask already has:
          - 3 channels: it returns the mask unchanged.
          - 4 channels: it returns only the first 3 channels,
          - less than 3 channels: it pads it with zeros to create an RGB representation.
    """

    def mask2rgb(self, max_value=1.0):
        shape = self.mask.shape

        if len(shape) == 2:
            mask = self.mask[:, :, np.newaxis]
        h, w, c = self.mask.shape
        if c == 3:
            return mask
        if c == 4:
            return self.mask[:, :, :3]

        if c > 4:
            raise ValueError

        padded = np.zeros((h, w, 3), dtype=self.mask.dtype)
        padded[:, :, :c] = self.mask
        padded = (padded * max_value).astype(np.uint8)

        return padded

    """sets specific color where mask value is equal to 1 (assuming that it is binary)"""

    def make_rgb_mask(self, color=(255, 0, 0)):
        h, w = self.mask.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[self.mask == 1.0, :] = color
        return rgb

    """
    this method overlays the mask over an image with a specific blending value alpha
    """

    def overlay_rgb_mask(self, sel, alpha):

        sel = sel == 1.0
        img2 = self.img
        img2[sel, :] = self.img[sel, :] * (1.0 - alpha) + self.mask[sel, :] * alpha
        return img2

    """this function takes a color map (cmap) and randomly selects colors from it for each instance
     and then overlays them over the image """

    def overlay_instances_mask(self, instances, cmap, alpha=0.9):
        h, w = self.img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        _max = instances.max()
        _cmax = cmap.shape[0]

        if _max == 0:
            return self.img
        elif _max > _cmax:
            indexes = [(i % _cmax) for i in range(_max)]
        else:
            indexes = random.sample(range(0, _cmax), _max)

        for i, idx in enumerate(indexes):
            overlay[instances == i + 1, :] = cmap[idx, :]

        overlay = (overlay * 255.0).astype(np.uint8)
        viz = self.overlay_rgb_mask(instances > 0, alpha=alpha)
        return viz
