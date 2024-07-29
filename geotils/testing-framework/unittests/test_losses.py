import unittest, torch, sys 
sys.path.append('../../')

from losses.losses import get_criterion, BCELoss, DiceLoss, FocalLoss, JaccardLoss, DiceBCELoss, JaccardBCELoss, DiceFocalLoss, JaccardFocalLoss, DiceFocalBCELoss, JaccardFocalBCELoss
from losses.lovasz import LovaszLoss, BinaryLovaszLoss


class TestGetCriterion(unittest.TestCase):
    def test_get_criterion_bce(self):
        criterion, _ = get_criterion(name='bce')
        self.assertIsInstance(criterion, BCELoss)
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    # def test_get_criterion_dice(self):
    #     criterion, _ = get_criterion(name='dice')
    #     self.assertIsInstance(criterion, DiceLoss)    
    #     y_pred = torch.rand(2, 1, 64, 64)
    #     y_gt = torch.rand(2, 1, 64, 64)
    #     loss = criterion(y_pred, y_gt)
    #     self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_focal(self):
        criterion, _ = get_criterion(name='focal')
        self.assertIsInstance(criterion, FocalLoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_jaccard(self):
        criterion, _ = get_criterion(name='jaccard')
        self.assertIsInstance(criterion, JaccardLoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_binary_lovasz(self):
        criterion, _ = get_criterion(name='blovasz')
        self.assertIsInstance(criterion, BinaryLovaszLoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_lovasz(self):
        criterion, _ = get_criterion(name='lovasz')
        self.assertIsInstance(criterion, LovaszLoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, (float, int))

    # def test_get_criterion_dice_bce(self):
    #     criterion, _ = get_criterion(name='dice+bce')
    #     self.assertIsInstance(criterion, DiceBCELoss)    
    #     y_pred = torch.rand(2, 1, 64, 64)
    #     y_gt = torch.rand(2, 1, 64, 64)
    #     loss = criterion(y_pred, y_gt)
    #     self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_jaccard_bce(self):
        criterion, _ = get_criterion(name='jaccard+bce')
        self.assertIsInstance(criterion, JaccardBCELoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    # def test_get_criterion_dice_focal(self):
    #     criterion, _ = get_criterion(name='dice+focal')
    #     self.assertIsInstance(criterion, DiceFocalLoss)    
    #     y_pred = torch.rand(2, 1, 64, 64)
    #     y_gt = torch.rand(2, 1, 64, 64)
    #     loss = criterion(y_pred, y_gt)
    #     self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_jaccard_focal(self):
        criterion, _ = get_criterion(name='jaccard+focal')
        self.assertIsInstance(criterion, JaccardFocalLoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    def test_get_criterion_jaccard_focal_bce(self):
        criterion, _ = get_criterion(name='jaccard+focal+bce')
        self.assertIsInstance(criterion, JaccardFocalBCELoss)        
        y_pred = torch.rand(2, 1, 64, 64)
        y_gt = torch.rand(2, 1, 64, 64)
        loss = criterion(y_pred, y_gt)
        self.assertIsInstance(loss, torch.Tensor)

    # def test_get_criterion_dice_focal_bce(self):
    #     criterion, _ = get_criterion(name='dice+focal+bce')
    #     self.assertIsInstance(criterion, DiceFocalBCELoss)    
    #     y_pred = torch.rand(2, 1, 64, 64)
    #     y_gt = torch.rand(2, 1, 64, 64)
    #     loss = criterion(y_pred, y_gt)
    #     self.assertIsInstance(loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
