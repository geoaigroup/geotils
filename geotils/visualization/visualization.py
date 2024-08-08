import numpy as np
import random
from skimage.draw import polygon


class Visualization(object):
    r"""Enter an image and mask to be able to modify them. All methods are static"""

    def mask2rgb(mask, max_value=1.0):
        r"""This method converts the mask to an RGB representation.
        If the mask already has:
        - 3 channels: it returns the mask unchanged.
        - 4 channels: it returns only the first 3 channels,
        - less than 3 channels: it pads it with zeros to create an RGB representation.

        Parameters
        ----------
        mask : np.ndarray
            Input mask array with shape (H, W).

        Returns
        -------
        np.ndarray
            rgb mask
        """
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

    def make_rgb_mask(mask, color=(255, 0, 0)):
        r"""Sets specific color where mask value is equal to 1 (assuming that it is binary)
        and returns the produced rgb mask

        Parameters
        ----------
        mask : np.ndarray
            Input mask array with shape (H, W).

        Returns
        -------
        np.ndarray
            rgb mask
        """

        h, w = mask.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask == 1.0, :] = color
        return rgb

    def overlay_rgb_mask(img, mask, sel, alpha):
        r"""This method overlays the mask over an image with a specific blending value alpha

        Parameters
        ----------
        img : np.ndarray
            Input image array with shape (H, W).
        mask : np.ndarray
            Input mask array with shape (H, W).
        sel: int
            decide
        alpha: int
            blending value

        Returns
        -------
        np.ndarray
            rgb mask
        """
        sel = sel == 1.0
        img[sel, :] = img[sel, :] * (1.0 - alpha) + mask[sel, :] * alpha
        return img

    def overlay_instances_mask(img, instances, cmap, alpha=0.9):
        """This function takes a color map (cmap) and randomly selects colors from it for each instance
        and then overlays them over the image
        Parameters
        ----------
        img : np.ndarray
            Input image array with shape (H, W).
        instances : np.ndarray
            Input mask array with shape (H, W).
        cmap: np.ndarray
            color map array
        alpha: int
            blending value

        Returns
        -------
        np.ndarray
            rgb mask
        """
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


def draw_polygon(geom,mask,value=255,zero_value=0):
    coords = np.asarray(list(geom.exterior.coords))
    rr,cc = polygon(coords[:,1],coords[:,0],shape=mask.shape[:2])
    mask[rr,cc] = value
    for lr in geom.interiors:
        interior_coords = np.asarray(list(lr.coords))
        rr,cc = polygon(interior_coords[:,1],interior_coords[:,0],shape=mask.shape[:2])
        mask[rr,cc] = zero_value
    return mask

def draw_shape(geom,mask,value=255,zero_value=0):
    if geom.geom_type == 'Polygon':
        polys = [geom]
    elif geom.geom_type == 'MultiPolygon':
        polys = list(geom.geoms)
    else:
        raise NotImplementedError

    for poly in polys:
        mask = draw_polygon(poly,mask,value,zero_value)
    
    return mask

def gdf_to_mask(gdf,mask_shape,binary=False):
    H,W = mask_shape
    mask = np.zeros((H,W),dtype=np.uint8 if binary else np.uint16)
    for i,row in gdf.iterrows():
        geom = row['geometry']
        index = 255 if binary else i+1 
        draw_shape(geom,mask,index,0)
    return mask