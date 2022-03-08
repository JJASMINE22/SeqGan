import torch
import torch.nn.functional as F
from models.generator import LSTMGenerator

class SeqGAN_G(LSTMGenerator):
    def __init__(self, 
                 **kwargs):
        super(SeqGAN_G, self).__init__(**kwargs)

    def batchPGLoss(self, sources, targets, reward):

        batch_size, seq_len = targets.size()

        hidden_state, cell_state = self.init_hidden(batch_size=batch_size)

        out, _ = self.forward(sources.permute([1, 0]), [hidden_state, cell_state])

        out = out.view(-1, batch_size, self.vocab_size).permute([1, 0, 2])

        # NLL loss
        onehot_targets = F.one_hot(targets, self.vocab_size).float()
        loss = torch.abs(torch.sum(onehot_targets*out, dim=-1))

        reward_loss = torch.sum(loss*reward)

        return reward_loss
