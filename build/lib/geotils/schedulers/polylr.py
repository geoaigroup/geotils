from torch.optim.lr_scheduler import LambdaLR
# polylr

class PolyLR_WWP(LambdaLR):
    def __init__(self, optimizer, epochs, warmup,ratio=0.9):
        warmup = min(epochs,max(0,warmup-1))
        decay_epochs = epochs - warmup
        xlambda = lambda x : 1.0 if(x<warmup) else (1 - ((x - warmup) / decay_epochs ) ** ratio)
        super().__init__(optimizer, xlambda)

class PolyLR(PolyLR_WWP):
    def __init__(self, optimizer, epochs, ratio=0.9):
        super().__init__(optimizer, epochs,0,ratio)