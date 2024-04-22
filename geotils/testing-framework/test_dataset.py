import unittest

from numpy import array
from torch import tensor
import torch
import sys


sys.path.append("../")
from dataset.examples.georeferneced.Massachusetts_Buildings_Dataset import (
    MassachusettsBuildingsDataset,
)

from dataset.examples.nongeoreferenced.vlm import vlm
from torchgeo.datasets import BoundingBox


class TestGetMethod(unittest.TestCase):
    def test_vlm(self):
        v = vlm("assets/data/vlm")
        ans = {
            "id": 0,
            "date_added": 1543570371.6343634,
            "question_id": 0,
            "people_id": 0,
            "answer": "urban",
            "active": True,
        }
        q = {
            "id": 0,
            "date_added": 1543570371.6343553,
            "img_id": 0,
            "people_id": 0,
            "type": "rural_urban",
            "question": "Is it a rural or an urban area",
            "answers_ids": [0],
            "active": True,
        }
        t = tensor(
            [
                49.0,
                48.0,
                49.0,
                50.0,
                52.0,
                50.0,
                55.0,
                48.0,
                44.0,
                40.0,
                31.0,
                28.0,
                26.0,
                27.0,
                30.0,
                32.0,
                32.0,
                31.0,
                32.0,
                32.0,
                32.0,
                32.0,
                31.0,
                33.0,
                32.0,
                33.0,
                33.0,
                35.0,
                41.0,
                46.0,
                48.0,
                48.0,
                50.0,
                50.0,
                51.0,
                46.0,
                42.0,
                41.0,
                40.0,
                39.0,
                35.0,
                33.0,
                33.0,
                33.0,
                33.0,
                31.0,
                31.0,
                33.0,
                32.0,
                33.0,
                34.0,
                32.0,
                33.0,
                33.0,
                39.0,
                44.0,
                44.0,
                42.0,
                41.0,
                48.0,
                50.0,
                49.0,
                39.0,
                32.0,
                27.0,
                26.0,
                26.0,
                28.0,
                29.0,
                29.0,
                28.0,
                28.0,
                27.0,
                28.0,
                29.0,
                29.0,
                34.0,
                38.0,
                50.0,
                68.0,
                58.0,
                49.0,
                45.0,
                46.0,
                80.0,
                79.0,
                67.0,
                55.0,
                60.0,
                74.0,
                61.0,
                72.0,
                76.0,
                60.0,
                59.0,
                71.0,
                72.0,
                57.0,
                62.0,
                60.0,
                43.0,
                36.0,
                45.0,
                52.0,
                46.0,
                47.0,
                44.0,
                49.0,
                52.0,
                70.0,
                70.0,
                49.0,
                81.0,
                87.0,
                62.0,
                62.0,
                58.0,
                55.0,
                63.0,
                56.0,
                41.0,
                39.0,
                40.0,
                43.0,
                48.0,
                51.0,
                41.0,
                41.0,
                61.0,
                60.0,
                58.0,
                54.0,
                50.0,
                47.0,
                34.0,
                52.0,
                88.0,
                107.0,
                95.0,
                95.0,
                100.0,
                76.0,
                77.0,
                76.0,
                74.0,
                97.0,
                85.0,
                44.0,
                57.0,
                84.0,
                88.0,
                78.0,
                59.0,
                56.0,
                87.0,
                96.0,
                59.0,
                42.0,
                55.0,
                53.0,
                46.0,
                67.0,
                65.0,
                60.0,
                71.0,
                91.0,
                85.0,
                66.0,
                59.0,
                61.0,
                58.0,
                47.0,
                55.0,
                62.0,
                56.0,
                47.0,
                49.0,
                61.0,
                65.0,
                69.0,
                88.0,
                68.0,
                82.0,
                67.0,
                43.0,
                49.0,
                49.0,
                37.0,
                43.0,
                70.0,
                69.0,
                82.0,
                77.0,
                93.0,
                71.0,
                57.0,
                65.0,
                52.0,
                47.0,
                48.0,
                76.0,
                83.0,
                70.0,
                65.0,
                65.0,
                63.0,
                80.0,
                93.0,
                91.0,
                96.0,
                98.0,
                84.0,
                61.0,
                48.0,
                72.0,
                66.0,
                63.0,
                65.0,
                59.0,
                59.0,
                60.0,
                65.0,
                61.0,
                53.0,
                47.0,
                69.0,
                68.0,
                49.0,
                64.0,
                85.0,
                48.0,
                51.0,
                66.0,
                72.0,
                62.0,
                50.0,
                56.0,
                59.0,
                53.0,
                47.0,
                47.0,
                49.0,
                51.0,
                60.0,
                58.0,
                70.0,
                55.0,
                61.0,
                57.0,
                65.0,
                51.0,
                50.0,
                54.0,
                72.0,
                78.0,
                73.0,
            ]
        )
        item = v.__getitem__(0)
        self.assertEqual(item["image"].data[0][0].all(), t.all())
        self.assertEqual(item["answers"][0], ans)
        self.assertEqual(item["questions"][0], q)

    def test_Massachusetts_building(self):

        bb = BoundingBox(
            225986.4361739957, 245486.43617399564, 883271.0247782532, 917771.025, 0, 9
        )
        mc = MassachusettsBuildingsDataset("assets/data/archive")
        to_test = to_test = torch.nonzero(
            mc.__getitem__(bb, mask_extension=".tif")["mask"]
        ).numpy()[0]
        self.assertEqual(
            to_test.all(),
            array([0, 18000, 1510]).all(),
        )


if __name__ == "__main__":
    unittest.main()