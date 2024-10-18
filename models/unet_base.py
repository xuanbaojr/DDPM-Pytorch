import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()

    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X):
        assert hasattr(self, 'net'), "Neural network is defined"
        return self.net(X)
    
    def training_step(self, noisy_pred, noise):
        print("noisy_pred.shape, noisy_im.shape", noisy_pred.shape, noise.shape)
        # plot figure of loss
        l = self.loss(noisy_pred, noise)
        return l
    
    def validation_step(self, batch):
        raise NotImplementedError
    
    def configure_optimizers(self):
        raise NotImplementedError

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xt, t):
        return xt
    
    def loss(self, y_hat, y):
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        return loss