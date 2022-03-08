# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
from torch import nn
from utils.rollout import ROLLOUT
from models.Seq_generator import SeqGAN_G
from models.Seq_discriminator import SeqGAN_D


class SeqGAN:
    def __init__(self,
                 gen_embed_dim: int,
                 gen_hidden_dim: int,
                 vocab_size: int,
                 max_seq_len: int,
                 dis_embed_dim: int,
                 filter_sizes: list,
                 num_filters: list,
                 Lambda: float,
                 gen_lr: float,
                 dis_lr: float,
                 batch_size: int,
                 rollout_num: int,
                 dropout: float,
                 padding_idx=None,
                 device=None,
                 ignore_pretrain=False,
                 gen_ckpt_path=None,
                 dis_ckpt_path=None,
                 **kwargs):
        self.gen_embed_dim = gen_embed_dim
        self.gen_hidden_dim = gen_hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dis_embed_dim = dis_embed_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.Lambda = Lambda
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.batch_size = batch_size
        self.rollout_num = rollout_num
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.device = device

        self.gen = SeqGAN_G(embed_dim=self.gen_embed_dim,
                            hidden_dim=self.gen_hidden_dim,
                            vocab_size=self.vocab_size,
                            max_seq_len=self.max_seq_len,
                            padding_idx=self.padding_idx)
        self.dis = SeqGAN_D(embed_dim=self.dis_embed_dim,
                            vocab_size=self.vocab_size,
                            filter_sizes=self.filter_sizes,
                            num_filters=self.num_filters,
                            padding_idx=self.padding_idx,
                            dropout=self.dropout)

        if self.device:
            self.gen = self.gen.to(self.device)
            self.dis = self.dis.to(self.device)
        if ignore_pretrain:
            gen_ckpt = torch.load(gen_ckpt_path)
            gen_state_dict = gen_ckpt['state_dict']
            dis_ckpt = torch.load(dis_ckpt_path)
            dis_state_dict = dis_ckpt['state_dict']
            self.gen.load_state_dict(gen_state_dict)
            self.dis.load_state_dict(dis_state_dict)

        self.rollout_func = ROLLOUT(self.gen)

        self.gen_opt = torch.optim.Adam(params=self.gen.parameters(), lr=self.gen_lr)
        self.dis_opt = torch.optim.Adam(params=self.dis.parameters(), lr=self.dis_lr)

        self.pre_gen_loss, self.adv_gen_loss = 0, 0
        self.dis_loss, self.dis_acc = 0, 0

    def optimize(self, loss, optimizer):

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def pretrain_generator(self, targets):

        targets = torch.LongTensor(targets)
        if self.device:
            targets = targets.to(self.device)

        _, predictions, _ = self.gen.forward_seqgan(targets)
        loss = self.gen.nll_loss(predictions.permute([0, 2, 1]), targets)

        # l2 regularization
        for weight in self.gen.weights:
            loss += self.Lambda*torch.sum(torch.square(weight))

        self.optimize(loss, self.gen_opt)
        self.pre_gen_loss += loss.data.item()

    def adv_train_generator(self):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        with torch.no_grad():
            gen_samples = self.gen.sample(num_samples=self.batch_size,
                                          batch_size=self.batch_size)
            targets = gen_samples
            sources = torch.zeros_like(targets)
            sources[:, 0] = cfg.start_letter
            sources[:, 1:] = targets[:, :self.max_seq_len-1]

        if self.device:
            targets = targets.to(self.device)
            sources = sources.to(self.device)

        rewards = self.rollout_func.get_reward(targets, self.rollout_num, self.dis)  #.cpu() reward with MC search
        loss = self.gen.batchPGLoss(sources, targets, rewards)

        # l2 regularization
        for weight in self.gen.weights:
            loss += self.Lambda * torch.sum(torch.square(weight))

        # update parameters
        self.optimize(loss, self.gen_opt)
        self.adv_gen_loss += loss.data.item()

    def train_discriminator(self, real_sources, fake_sources):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        real_sources = torch.LongTensor(real_sources)
        fake_sources = torch.LongTensor(fake_sources)
        if self.device:
            real_sources = real_sources.to(self.device)
            fake_sources = fake_sources.to(self.device)

        real_predictions = self.dis.forward(real_sources)
        fake_predictions = self.dis.forward(fake_sources)
        predictions = torch.cat((real_predictions, fake_predictions), dim=0)
        labels = torch.cat((torch.ones_like(real_predictions[:, 0], dtype=torch.long),
                            torch.zeros_like(fake_predictions[:, 0], dtype=torch.long)), dim=0) \
            if cfg.loss_mode == 'CrossEntropy' else \
            torch.cat((torch.ones_like(real_predictions),
                       -torch.ones_like(fake_predictions)), dim=0)
        loss = self.dis.discriminator_loss(predictions, labels)

        # l2 regularization
        for weight in self.dis.weights:
            loss += self.Lambda * torch.sum(torch.square(weight))

        self.optimize(loss, self.dis_opt)

        self.dis_loss += loss.data.item()
        self.dis_acc += (torch.sum(torch.gt(real_predictions.squeeze(1), 0)).data.item() +
                         torch.sum(torch.less_equal(fake_predictions.squeeze(1), 0)).data.item())/predictions.size(0)
        # self.dis_acc += torch.sum((torch.eq(predictions.argmax(dim=-1), labels))).data.item()/predictions.size(0)
