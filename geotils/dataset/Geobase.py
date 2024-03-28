import torch
import sys


from dataset.Geotorch_datasets import geo
from torchgeo.datasets import BoundingBox


"""Base class for all geospatial datasets."""
import os
import re
from typing import Any, cast


import numpy as np
import rasterio
import rasterio.merge
import torch
from torch import Tensor


class GeotilsRasterDataset(geo.RasterDataset):
    def __getitem__(self, query: BoundingBox, mask_extension=".tiff") -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "band" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bbox": query}

        data = data.to(self.dtype)
        if self.is_image:
            sample["image"] = data
        else:

            sample["image"] = data
            maskpaths = []

            for path in filepaths:
                maskpaths.append(
                    (
                        os.path.join(
                            (os.path.split(path)[0] + "_labels"),
                            os.path.split(path)[1][0:-5] + mask_extension,
                        ),
                        path,
                    )
                )
            masks = []
            for filepath in maskpaths:
                original = filepath[1]
                target = filepath[0]
                with rasterio.open(original) as src:
                    source_profile = src.profile
                    source_transform = src.transform
                    source_crs = self.crs
                    src.close()

                dst = rasterio.open(target, "r+")

                dst.crs = source_crs
                dst.transform = source_transform

                masks.append(dst)

            bounds = (query.minx, query.miny, query.maxx, query.maxy)
            dest, _ = rasterio.merge.merge(
                masks, bounds, self.res, indexes=self.band_indexes
            )
            # fix numpy dtypes which are not supported by pytorch tensors
            if dest.dtype == np.uint16:
                dest = dest.astype(np.int32)
            elif dest.dtype == np.uint32:
                dest = dest.astype(np.int64)
            data = torch.tensor(dest)
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
