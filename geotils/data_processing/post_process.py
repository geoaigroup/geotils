import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import shapes
from skimage.morphology import dilation,square,erosion
from skimage.segmentation import watershed
from skimage.measure import label
from PIL import Image,ImageDraw
import pandas as pd
from shapely.geometry import shape
from shapely.wkt import dumps
from shapely.ops import unary_union


class PostProcessing():
    def post_process(self, pred, thresh=0.5, thresh_b=0.6, mina=100, mina_b=50):
        """
        Post-processes a prediction mask to obtain a refined segmentation.This function refines 
        a semantic segmentation mask, particularly for building segmentation tasks.
        It leverages optional channels for predicting borders and spacing around buildings.

        Parameters:
            pred (numpy.ndarray): Prediction mask with shape (height, width, channels).
            thresh (float, default=0.5): Threshold for considering pixels as part of the final segmentation.
            thresh_b (float, default=0.6): Threshold for considering pixels as borders between objects.
            mina (int, default=100): Minimum area threshold for retaining segmented regions.
            mina_b (int, default=50): Minimum area threshold for retaining basins.

        Returns:
            numpy.ndarray: Refined segmentation mask.

        Description:
            The refinement process involves:
            1. Extracting individual channels from the input mask.
            2. Separating nuclei (building interiors) from borders based on the predicted borders channel.
            3. Applying thresholding to identify basins within nuclei, which represent potential individual buildings.
            4. Optionally filtering out small basins based on the minimum area threshold.
            5. Performing watershed segmentation to separate closely located buildings.
            6. Applying noise filtering to remove small regions from the segmented mask.
            7. Returning the refined segmentation mask with labeled and filtered regions.

        Note:
            - The function assumes a specific order for input channels:
                - Channel 0: Building predictions
                - Channel 1 (optional): Border predictions
                - Channel 2 (optional): Spacing predictions
            - The output represents labeled regions in the refined segmentation.

        """
        if len(pred.shape) < 2:
            return None
        if len(pred.shape) == 2:
            pred = pred[..., np.newaxis]

        ch = pred.shape[2]
        buildings = pred[..., 0]

        if ch > 1:
            borders = pred[..., 1]
            nuclei = buildings * (1.0 - borders)

            if ch == 3:
                spacing = pred[..., 2]
                nuclei *= (1.0 - spacing)

            basins = label(nuclei > thresh_b, background=0, connectivity=2)

            if mina_b > 0:
                basins = self.noise_filter(basins, mina=mina_b)
                basins = label(basins, background=0, connectivity=2)

            washed = watershed(image=-buildings, markers=basins, mask=buildings > thresh, watershed_line=False)

        elif ch == 1:
            washed = buildings > thresh

        washed = label(washed, background=0, connectivity=2)
        washed = self.noise_filter(washed, mina=mina)
        washed = label(washed, background=0, connectivity=2)

        return washed
    
    @staticmethod
    def noise_filter(washed,mina):
        """
        Filter small regions in a labeled segmentation mask based on minimum area.
        This function filters out small labeled regions in a segmentation mask based on their area.
        It iterates over unique labels, calculates the area for each label, and sets the pixel values
        corresponding to labels with area less than or equal to the specified threshold to 0.

        Args:
            washed (numpy.ndarray): Input labeled segmentation mask.
            mina (int): Minimum area threshold for retaining labeled regions.

        Returns:
            numpy.ndarray: Segmentation mask with small labeled regions filtered out.
        """
        values = np.unique(washed)
        for val in values[1:]:
            area = (washed[washed == val]>0).sum()
            if area<=mina:
                washed[washed == val] = 0
        return washed

    @staticmethod
    def extract_poly(mask):
        """
        Extract polygons from a binary mask using the `rasterio.features.shapes` method.
        It then processes each shape, converts it to a Shapely Polygon, and appends it to a list.
        The list of polygons is then combined into a single geometry using `shapely.ops.unary_union`.
        The resulting polygon or None (if no polygons are found) is returned.

        Args:
            mask (numpy.ndarray): Binary input mask.

        Returns:
            shapely.geometry.base.BaseGeometry or None: Resulting polygon or None if no polygons are found.
        """
        shps = shapes(mask.astype(np.int16), mask > 0)
        polys = []
        #check for validity to avoid crashes
        try:
            for shp, _ in shps:
                p = shape(shp).buffer(0.0)
                typ = p.geom_type
                if typ == 'Polygon' or typ == 'MultiPolygon':
                    polys.append(p.simplify(0.01))
                else:
                    continue
        except:
            return None

        if len(polys) == 0:
            return None
        else:
            return unary_union(polys)
        
    @staticmethod    
    def instance_mask_to_gdf(instance_mask, transform=None, crs=None):
        """
        Convert an instance mask to a GeoDataFrame, uses rasterio's `shapes` method to obtain shapes from the instance mask.
        The shapes are then converted to a GeoDataFrame, where each geometry is associated with a unique id.
        The resulting GeoDataFrame is dissolved by the 'id' column to form a single geometry for each instance.
        If no instances are found, an empty GeoDataFrame is returned.

        Args:
            instance_mask (numpy.ndarray): Input instance mask with shape (H, W), where each instance is labeled by a unique id/number.
            transform (affine.Affine, Optional): Geospatial transform of the raster. Default is None.
            crs (str, Optional): CRS of the raster. Default is None.

        Returns:
            geopandas.GeoDataFrame: GeoDataFrame of the shapes projected to the specified CRS using the transform.
        """
        # Transform should be Identity if None is provided
        transform = rio.transform.IDENTITY if transform is None else transform

        all_shapes = shapes(instance_mask, mask=None, transform=transform)
        data = [{'properties': {'id': v}, 'geometry': s} for i, (s, v) in enumerate(all_shapes) if v != 0]

        if len(data) == 0:
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs=crs)

        gdf = gpd.GeoDataFrame.from_features(data, crs=crs)
        gdf = gdf.dissolve(by='id')

        return gdf
