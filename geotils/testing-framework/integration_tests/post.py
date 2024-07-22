from geotils.data_processing.post_process import PostProcessing
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd


class test_postproccesing:
    height = 200
    width = 200
    global mask
    mask = np.zeros((height, width), dtype=np.uint8)

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    cx = width // 2
    cy = height // 2
    r_inner = 20
    r_outer = 50

    star_mask = (
        ((X - cx) ** 2 + (Y - cy) ** 2 <= r_outer**2)
        & ((X - cx) ** 2 + (Y - cy) ** 2 >= r_inner**2)
        & (
            (X - cx) ** 2 + (Y - cy) ** 2
            <= r_outer**2 - 2 * (r_outer - r_inner) * np.abs(X - cx)
        )
    )

    mask[star_mask] = 1
    global post
    post = PostProcessing()

    def test_extract_poly():
        mask2 = post.extract_poly(mask)
        x, y = mask2.exterior.xy

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(mask)
        gpd.GeoSeries(mask2).plot(ax=axs[1])
        plt.show()

    def test_instance_mask_to_gdf():
        height = 200
        width = 200

        instance_mask = np.zeros((height, width), dtype=np.uint8)

        cy1, cx1 = 50, 50
        radius1 = 30
        Y, X = np.ogrid[:height, :width]
        mask1 = (X - cx1) ** 2 + (Y - cy1) ** 2 <= radius1**2
        instance_mask[mask1] = 1

        cy2, cx2 = 150, 150
        a, b = 40, 20
        mask2 = ((X - cx2) ** 2 / a**2 + (Y - cy2) ** 2 / b**2) <= 1
        instance_mask[mask2] = 2

        post.instance_mask_to_gdf(instance_mask).plot()
