from Geotorch_datasets import geo
from torchgeo.datasets import BoundingBox
from examples.georeferneced.Massachusetts_Buildings_Dataset import (
    MassachusettsBuildingsDataset,
)


class GeotilsRasterDataset(geo.RasterDataset):
    def __getitem__(self, query: BoundingBox):
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
                    os.path.join(
                        (os.path.split(path)[0] + "_labels"), os.path.split(path)[1]
                    )
                )
            data = self._merge_files(filepaths, query, self.band_indexes)
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def test_Massachusetts_building():
    bb = BoundingBox(
        225986.4361739957, 245486.43617399564, 883271.0247782532, 917771.025, 0, 9
    )
    mc = MassachusettsBuildingsDataset(
        r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotorch\archive"
    )
    to_test = mc.__getitem__(bb, mask_extension=".tif")

    print()


test_Massachusetts_building()
