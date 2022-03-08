import torch
import config as cfg
from models.discriminator import CNNDiscriminator

class SeqGAN_D(CNNDiscriminator):
    def __init__(self,
                 **kwargs):
        super(SeqGAN_D, self).__init__(**kwargs)

    def discriminator_loss(self, preds, labels):

        if cfg.loss_mode == 'CrossEntropy':
            loss = self.cross_entropy_loss(preds, labels)
        else:
            loss = torch.relu(1. - torch.mul(preds,labels)).mean()

        return loss
