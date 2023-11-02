#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:39:43 2020

@author: hasan
"""
import torch
import math
import torch.optim
from torch.optim import SGD,Adam,RMSprop
from Training.AdamW import AdamW
from torch.optim.lr_scheduler import CyclicLR,OneCycleLR,ReduceLROnPlateau,CosineAnnealingLR,CosineAnnealingWarmRestarts

optimizers = ['Adam','SGD','AdamW','RMSprop','RAdam']
schedulers = ['CyclicLR','OneCycleLR','ReduceLROnPlateau','MultiStepLR','StepLR']

def optimizer_maker(parameters ,name = 'Adam', beta=(0.9,0.99), epsilon=1e-8, wd=1e-4 , learning_rate=0.0001, moment = (0.9,0), alfa =0.99):
    assert(name in optimizers)
    if(name == optimizers[0]):
        optimizer = Adam(params = parameters,
                         lr=learning_rate,
                         betas=beta,
                         eps=epsilon,
                         weight_decay=wd,
                         amsgrad=False)
    elif(name == optimizers[1]):
        optimizer = SGD(params = parameters,
                        lr=learning_rate,
                        momentum = moment[0],
                        dampening=0,
                        weight_decay=wd,
                        nesterov=True)
    elif(name == optimizers[2]):
        optimizer = AdamW(params = parameters,
                          lr= learning_rate,
                          betas= beta,
                          eps= epsilon,
                          weight_decay=wd)
    elif(name == optimizers[3]):
        optimizer = RMSprop(params = parameters,
                            lr=learning_rate,
                            alpha = alfa,
                            eps = epsilon,
                            weight_decay= wd,
                            momentum= moment[1],
                            centered=False)
    elif(name == optimizers[4]):
        optimizer = RAdam(params = parameters,
                         lr=learning_rate,
                         betas=beta,
                         eps=epsilon,
                         weight_decay=wd)
                         #amsgrad=False)
        
    return optimizer

def get_scheduler(optimizer,epochs=50,step=50*12,
                  name='polylr',
                  lr=0.0001,
                  min_lr=0.0001/4,
                  ratio=0.5,
                  patience=3,
                  thresh=0.01,
                  cool=0):
    rate_map = {'reduceonplateau':'per_epoch_withmetric',
               'polylr':'per_epoch',
               'cycliclr':'per_batch',
               'onecyclelr': 'per_batch',
               'cosannealinglrwm' : 'per_batch_withinfo',
               'cosannealinglr' : 'per_epoch',
               'ipolylr'        : 'per_batch_report'
               
               }
    rate = lambda x : rate_map[x.lower()]
    if(name.lower()=='reduceonplateau'):
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',
                                      factor=ratio,
                                      patience=patience,
                                      verbose=True,
                                      threshold=thresh,
                                      threshold_mode='rel',
                                      cooldown=cool,
                                      min_lr=min_lr,
                                      eps=1e-08)
    elif(name.lower()=='polylr'):
        scheduler = PolyLR(optimizer,
                           epochs,
                           ratio=ratio)
    elif(name.lower()=='cycliclr'):
        scheduler = CyclicLR(optimizer,
                             base_lr=min_lr,
                             max_lr=lr,
                             step_size_up=step,
                             step_size_down=None,
                             mode='exp_range',
                             gamma=1.0,
                             scale_fn=None,
                             scale_mode='cycle',
                             cycle_momentum=True,
                             base_momentum=0.8,
                             max_momentum=0.9,
                             last_epoch=-1)
    elif(name.lower()=='onecyclelr'):
        scheduler = OneCycleLR(optimizer,
                               max_lr=lr,
                               total_steps=step,
                               pct_start=ratio,
                               anneal_strategy='cos',
                               cycle_momentum=True,
                               base_momentum=0.85,
                               max_momentum=0.95,
                               div_factor=20.0, 
                               final_div_factor=10000.0,
                               last_epoch=-1)
    elif(name.lower() == 'cosannealinglr'):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=step,
                                      eta_min=1e-6,
                                      last_epoch=-1)
    elif(name.lower() == 'cosannealinglrwm'):
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0 = step,
                                                T_mult = ratio,
                                                eta_min=1e-6,
                                                last_epoch=-1)
    elif(name.lower()=='ipolylr'):
        scheduler = Inverse_PolyLR(optimizer,
                           iterrs = step,
                           ratio=ratio)
    else:
        raise NotImplementedError
    return scheduler,rate(name.lower())
###Costum Optimizers and schedulers###

class Inverse_PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, iterrs,ratio=0.9):
        super().__init__(optimizer, lambda iterr: (iterr / iterrs) ** ratio)

class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs,ratio=0.9):
        super().__init__(optimizer, lambda epoch: (1 - (epoch / epochs) ** ratio))
########################################
from torch.optim.lr_scheduler import _LRScheduler
class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.t_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr  
########################################               
class RAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdamWarmup(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamWarmup, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWarmup, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss