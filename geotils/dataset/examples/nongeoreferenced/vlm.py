import json
from typing import Any, Callable, Optional
import torchgeo.datasets.geo as geo
import os
from PIL import Image
import numpy as np
import torch

"""link to used dataset https://zenodo.org/records/6344334"""


class vlm(geo.NonGeoDataset):
    """
    root : path to the data
    split : train / test
    transforms : function to transform data(optional)
    to each image an answer and a question is assigned and all of these are present in two files specified below
    (all_answers.json/all_questions.json)
    """

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

    """function that takes in the directory of the image , and it's id and returns it as a tensor """

    def _load_image(self, directory: str, id):
        filename = os.path.join(directory, "{}.tif".format(id))
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            tensor = tensor.permute((2, 0, 1))
            tensor = tensor.float()
            return tensor

