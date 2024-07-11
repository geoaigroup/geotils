import geotils.data_processing.augmentation as aug

import torch


class test_image:
    rotate = aug.TorchRandomRotate((1, 2, 3, 4, 4, 5, 6, 67))

    mask_tensor = torch.ones((1, 1, 256, 256))
    modified_mask = aug.RandomMaskIgnore(proba=0.8, max_length=256, min_length=256)(
        mask_tensor.clone()
    )
