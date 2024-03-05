import unittest

from torch import tensor
import torch
from Massachusetts_Buildings_Dataset import MassachusettsBuildingsDataset

# from vlm import vlm
from torchgeo.datasets import BoundingBox


class TestGetMethod(unittest.TestCase):
    # def test_vlm(self):
    #    v = vlm()
    #    self.assertEqual(vlm.__getitem__(), None)

    def test_Massachusetts_building(self):
        mask = tensor(
            [
                [
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 62, 67, 46],
                    [0, 0, 0, ..., 62, 42, 50],
                    [0, 0, 0, ..., 52, 78, 67],
                ],
                [
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 69, 70, 54],
                    [0, 0, 0, ..., 73, 40, 46],
                    [0, 0, 0, ..., 51, 83, 73],
                ],
                [
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 48, 58, 37],
                    [0, 0, 0, ..., 45, 32, 32],
                    [0, 0, 0, ..., 40, 66, 49],
                ],
            ],
            dtype=torch.uint8,
        )

        bb = BoundingBox(
            225986.4361739957, 245486.43617399564, 883271.0247782532, 917771.025, 0, 9
        )
        mc = MassachusettsBuildingsDataset(
            r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotorch\archive"
        )
        self.assertEqual(mc.__getitem__(bb).mask.all(), mask)


if __name__ == "__main__":
    unittest.main()
