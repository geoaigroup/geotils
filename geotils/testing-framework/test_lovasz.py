import unittest
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import sys 
sys.path.append('../')
from losses.lovasz import BinaryLovaszLoss, LovaszLoss, _lovasz_hinge, _lovasz_softmax, _lovasz_softmax_flat, _flatten_binary_scores, _flatten_probas, _lovasz_grad, _lovasz_hinge_flat

class TestLovaszLosses(unittest.TestCase):

    def setUp(self):
        self.logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        self.labels = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        self.logits_multiclass = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]]])
        self.labels_multiclass = torch.tensor([[[0, 1], [2, 0]]])

    def test_lovasz_grad(self):
        gt_sorted = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        grad = _lovasz_grad(gt_sorted)
        expected_grad = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)
        np.testing.assert_almost_equal(grad.numpy(), expected_grad.numpy())

    def test_binary_lovasz_hinge(self):
        loss_func = BinaryLovaszLoss()
        loss = loss_func(self.logits, self.labels)
        self.assertIsInstance(loss.item(), float)

    def test_lovasz_softmax(self):
        loss_func = LovaszLoss()
        loss = loss_func(self.logits_multiclass, self.labels_multiclass)
        self.assertIsInstance(loss.item(), float)

    def test_flatten_binary_scores(self):
        scores, labels = _flatten_binary_scores(self.logits, self.labels)
        self.assertEqual(scores.shape, labels.shape)

    def test_flatten_probas(self):
        probas, labels = _flatten_probas(self.logits_multiclass, self.labels_multiclass)
        self.assertEqual(probas.shape[0], labels.shape[0])

    def test_lovasz_hinge_flat(self):
        logits = torch.tensor([0.5, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0])
        loss = _lovasz_hinge_flat(logits, labels)
        self.assertIsInstance(loss.item(), float)

    def test_lovasz_softmax_flat(self):
        probas = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        labels = torch.tensor([1, 0, 1, 0])
        loss = _lovasz_softmax_flat(probas, labels)
        self.assertIsInstance(loss.item(), float)

if __name__ == '__main__':
    unittest.main()
