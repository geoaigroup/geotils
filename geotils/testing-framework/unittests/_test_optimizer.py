import unittest, sys, os
# sys.path.append('../data_processing')
import torch
from torch import nn
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))

sys.path.append(parent_directory)

from optimizers import get_optimizer,optimizer_mapping
import numpy as np


class TestOptimizers(unittest.TestCase):
    def test_getoptimzer_return(self):
        for optimizer in optimizer_mapping:
            net=nn.Sequential(
                nn.Linear(2,1),
            )
            optim=get_optimizer(optimizer,net.parameters(),lr=1e-3)
            optim.zero_grad()
            optim.step()
            self.assertIsInstance(optim,torch.optim.Optimizer,f"{optimizer} did not return an torch.optim.Optimizer")

    def test_getoptimzer_lookAhead(self):
        for optimizer in optimizer_mapping:
            net=nn.Sequential(
                nn.Linear(2,1),
            )
            if optimizer =="adan": #TODO fix adan
                with self.assertRaises(AssertionError) as context:
                    optim=get_optimizer(optimizer,net.parameters(),lr=1e-3,lookAhead=True)
            else:       
                optim=get_optimizer(optimizer,net.parameters(),lr=1e-3,lookAhead=True)
                optim.zero_grad()
                optim.step()

                self.assertIsInstance(optim,torch.optim.Optimizer,f"{optimizer} did not return an torch.optim.Optimizer")

    def test_keys_lower_case(self):
        for optim in optimizer_mapping:
            self.assertEqual(optim,optim.lower())

    def test_getoptimzer_functionality(self):
        for optimizer in optimizer_mapping:
            for __ in range(5):
                net=nn.Sequential(
                nn.Linear(2,1),
                )
                optim=get_optimizer(optimizer,net.parameters(),lr=1e-3)
                loss1=[]
                for _ in range(500):
                    optim.zero_grad()
                    X=torch.randint(0,10,(50,2),dtype=torch.float)
                    y=X.sum(axis=1)
                    loss=nn.MSELoss()(y,net(X))
                    loss.backward()
                    loss1.append(loss.item())
                    optim.step()
                loss2=[]
                for _ in range(500):
                    optim.zero_grad()
                    X=torch.randint(0,10,(50,2),dtype=torch.float)
                    y=X.sum(axis=1)
                    loss=nn.MSELoss()(y,net(X))
                    loss.backward()
                    loss2.append(loss.item())
                    optim.step()
                self.assertGreaterEqual(np.array(loss1).mean(),np.array(loss2).mean(),f"{optimizer} is not functioning as an optimizer")



if __name__=="__main__":
    unittest.main()
