import numpy as np
import random


class Visualization(object):
    """enter an image and mask to be able to modify them
    the img and mask will  be themselves modified and a modified version will be returned after every call
    """

    """This method converts the mask to an RGB representation.
        If the mask already has:
          - 3 channels: it returns the mask unchanged.
          - 4 channels: it returns only the first 3 channels,
          - less than 3 channels: it pads it with zeros to create an RGB representation.
    """

    def mask2rgb(mask, max_value=1.0):
        shape = mask.shape
        if len(shape) == 2:
            mask = mask[:, :, np.newaxis]
        h, w, c = mask.shape
        if c == 3:
            return mask
        if c == 4:
            return mask[:, :, :3]

        if c > 4:
            raise ValueError

        padded = np.zeros((h, w, 3), dtype=mask.dtype)
        padded[:, :, :c] = mask
        padded = (padded * max_value).astype(np.uint8)

        return padded

    """sets specific color where mask value is equal to 1 (assuming that it is binary)"""

    def make_rgb_mask(mask, color=(255, 0, 0)):
        h, w = mask.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask == 1.0, :] = color
        return rgb

    """
    this method overlays the mask over an image with a specific blending value alpha
    """

    def overlay_rgb_mask(img, mask, sel, alpha):

        sel = sel == 1.0
        img[sel, :] = img[sel, :] * (1.0 - alpha) + mask[sel, :] * alpha
        return img

    """this function takes a color map (cmap) and randomly selects colors from it for each instance
     and then overlays them over the image """

    def overlay_instances_mask(img, instances, cmap, alpha=0.9):
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        _max = instances.max()
        _cmax = cmap.shape[0]

        if _max == 0:
            return img
        elif _max > _cmax:
            indexes = [(i % _cmax) for i in range(_max)]
        else:
            indexes = random.sample(range(0, _cmax), _max)

        for i, idx in enumerate(indexes):
            overlay[instances == i + 1, :] = cmap[idx, :]

        overlay = (overlay * 255.0).astype(np.uint8)
        viz = Visualization.overlay_rgb_mask(img, overlay, instances > 0, alpha=alpha)
        return viz
