from torchgeo.datasets import geo
from torchgeo.datasets.geo import GeoDataset
import matplotlib.pyplot as plt


import numpy as np


from torchvision.datasets.folder import default_loader as pil_loader

"""this is an example of how to make a new geospatial(georefernced) dataset class using the geotorch RasterDataset class
for refrence check the README
the class implemented here is Massachusetts Buildings Dataset
"""


class MassachusettsBuildingsDataset(geo.RasterDataset):
    filename_glob = "*_15.tiff"
    is_image = False
    separate_files = False


# Correlation matrix
def plotCorrelationMatrix(dfe, graphWidth):
    filename = dfe.dataframeName
    dfe = dfe.dropna("columns")  # drop columns with NaN
    dfe = dfe[
        [col for col in dfe if dfe[col].nunique() > 1]
    ]  # keep columns where there are more than 1 unique values
    if dfe.shape[1] < 2:
        print(
            f"No correlation plots shown: The number of non-NaN or constant columns ({dfe.shape[1]}) is less than 2"
        )
        return
    corr = dfe.corr()
    plt.figure(
        num=None,
        figsize=(graphWidth, graphWidth),
        dpi=80,
        facecolor="w",
        edgecolor="k",
    )
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f"Correlation Matrix for {filename}", fontsize=15)
    plt.show()
