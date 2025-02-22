import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = {'lr':lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.params_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1

        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, betas, eps, weight_decay = group['lr'], group['betas'], group['eps'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get('t', 0)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))

                grad = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1])* (grad ** 2)

                lr_t = lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))

                p.data -= lr_t * m / (torch.sqrt(v) + eps) 
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss
    
