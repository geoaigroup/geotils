import unittest
import torch

from __init__ import get_optimizer,optimizer_mapping


class test_optimizers(unittest.TestCase):
    def test_getoptimzer_return(self):
        for optimizer in optimizer_mapping:
            net=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.assertIs(get_optimizer(optimizer,net.parameters()),torch.optim.Optimizer,f"{optimizer} did not return an optimizer")
        

if __name__=="__main__":
    test_optimizers().main()
