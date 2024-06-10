Torchgeo Dataset



Torchgeo documentation
This documentation will guide you through :
1- Using an existing dataset.
2- adding your own using the available classes


1- Using the existing datasets:


All the datasets that are available from torchgeo can be imported directly from the geotils library as follows :
you can directly import using from geotils import S1 (S1 name is temporary) then you can use the dataset that you want from S1 example :  sentinel1 = S1.Sentinel1(r"path/to/dataset")
if you want to add a dataset that has been added to torchgeo recently you can open the S1 file where you will find a group of imports as follows :

and just add the name of the dataset to the given list of imports(your torchgeo version should be up to date)


2- adding your own dataset :


torchgeo provides the following hagiarchy in classes(all of these can be found in geo.py):


Geodatasets(GeoDataset is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection.):
1 - RasterDataset
2 - VectorDataset
NonGeoDataset(non georeferenced):
abstract class

all of these classes inhert from the pytorch dataset class

these datasets have the advantage of having the ability to take their intersection or union using either the respective classes(intersectionDataset , UnionDataset) or using __and__ and __or__

To add your own dataset :

-First find type you need(as mentioned above).

if it is NonGeoDataset

you will need to implement the abstract methods , one example of this is a vlm dataset this is an example of the implementation of a vlm dataset:

import json
from typing import Any, Callable, Optional
import torchgeo.datasets.geo as geo
import os
from PIL import Image
import numpy as np
import torch


class vlm(geo.NonGeoDataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        answers_path = os.path.join(root, "all_answers.json")
        questions_path = os.path.join(root, "all_questions.json")

        with open(answers_path) as f:
            self.answers = json.load(f)["answers"]

        with open(questions_path) as f:
            self.questions = json.load(f)["questions"]

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, index: int) -> dict[str, Any]:
        directory = os.path.join(self.root, "Images_LR")
        sample: dict[str, Any] = {"image": self._load_image(directory, index)}
        features = {"answers": self.answers, "questions": self.questions}
        sample.update(features)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, directory: str, id):
        filename = os.path.join(directory, "{}.tif".format(id))
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            tensor = tensor.permute((2, 0, 1))
            tensor = tensor.float()
            return tensor


testVQA = vlm(r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotorch")


the main methods are __init__() and get_item() , the implementation of such a dataset will depend on your use case here we are adding each answer and question to a list (each question and answer is related to an image) and in the get_item() method we are returning the images info(tensor) based on the index image’s information(tensor) and its mask’s info to

more formally :


__init__(root='data', transforms=None, download=False, checksum=False)
Initialize a new ADVANCE dataset instance.
Parameters:

* root (str) – root directory where dataset can be found
* transforms (Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]]) – a function/transform that takes input sample and its target as entry and returns a transformed version
* download (bool) – if True, download dataset and store it in the root directory
* checksum (bool) – if True, check the MD5 of the downloaded files (may be slow)

Raises:
DatasetNotFoundError – If dataset is not found and download is False.
__getitem__(index)
Return an index within the dataset.
Parameters:
index (int) – index to return
Returns:
data and label at that index
Return type:
dict[str, torch.Tensor]




Adding a Geodataset :


it is different as alot of the implementation is already there you might only need to specify some of the given variables or add some of your own.

I will give two examples one that we have added and one that was already included in torchgeo:

The one that we have added is the Massachusetts Building segementation dataset


class MC(geo.RasterDataset):
    filename_glob = "*_15.tiff"
    is_image = False
    separate_files = False

(Yes this simple!!)

filename_glob:
Glob expression used to search for files.
This expression should be specific enough that it will not pick up files from other datasets. It should not include a file extension, as the dataset may be in a different file format than what it was originally downloaded as.

filename_regex:
Regular expression used to extract date from filename.
The expression should use named groups. The expression may contain any number of groups. The following groups are specifically searched for by the base class:

    * date: used to calculate mint and maxt for index insertion
    * start: used to calculate mint for index insertion
    * stop: used to calculate maxt for index insertion

When separate_files is True, the following additional groups are searched for to find other files:

    * band: replaced with requested band name

is_image = FALSE
True if dataset contains imagery, False if dataset contains mask(originally you could only have one but now if it is false you will have both)

some common functions:

__init__(paths='data', crs=None, res=None, transforms=None, download=False, cache=True)
Initialize a new Dataset instance.
Parameters:

* paths (Union[str, Iterable[str]]) – one or more root directories to search or files to load
* crs (Optional[CRS]) – coordinate reference system (CRS) to warp to (defaults to the CRS of the first file found)
* res (Optional[float]) – resolution of the dataset in units of CRS (defaults to the resolution of the first file found)
* transforms (Optional[Callable[[dict[str, Any]], dict[str, Any]]]) – a function/transform that takes an input sample and returns a transformed version
* download (bool) – if True, download dataset and store it in the root directory
* cache (bool) – if True, cache file handle to speed up repeated sampling

Raises:
DatasetNotFoundError – If dataset is not found and download is False.

plot(sample, show_titles=True, suptitle=None)[SOURCE]
Plot a sample from the dataset.
Parameters:

* sample (dict[str, Any]) – a sample returned by RasterDataset.__getitem__()
* show_titles (bool) – flag indicating whether to show titles above each panel
* suptitle (Optional[str]) – optional string to use as a suptitle

Returns:
a matplotlib Figure with the rendered sample
Return type:
Figure
you can also add your own plot function:

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



Aboveground Woody Biomass(dataset that was already present):


is_image = FALSE

filename_glob = '*N_*E.*'

filename_regex = '^\N  (?P<LATITUDE>[0-9][0-9][A-Z])_\N  (?P<LONGITUDE>[0-9][0-9][0-9][A-Z])*\N  '


Airphen(dataset that was already present):



* 6 Synchronized global shutter sensors
* Sensor resolution 1280 x 960 pixels
* Data format (.tiff, 12 bit)
* SD card storage
* Metadata information: Exif and XMP
* Internal or external GPS
* Synchronization with different sensors (TIR, RGB, others)



all_bands = LIST[STR] = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
rgb_bands = LIST[STR] = ['B4', 'B3', 'B1']
