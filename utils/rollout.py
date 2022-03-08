# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import copy
import torch
import config as cfg

class ROLLOUT:
    def __init__(self, gen):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size

    def get_reward(self, sentences, rollout_num, discriminator, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param discriminator:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size], dtype=torch.float)
            if cfg.device:
                rewards = rewards.to(cfg.device)
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = discriminator(samples)
                    out = torch.softmax(out, dim=-1) if cfg.loss_mode is 'CrossEntropy' else torch.sigmoid(out)
                    reward = out[:, current_k+1] if cfg.loss_mode is 'CrossEntropy' else out[:, current_k]
                    rewards[idx] = reward
                    idx += 1

        # rewards = torch.mean(rewards, dim=0)
        # rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        rewards = rewards.view(rollout_num, self.max_seq_len, batch_size).mean(dim=0).permute([1, 0])

        return rewards

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size=batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num].permute([1, 0])
        out, hidden = self.gen.forward(inp, hidden)

        out = out.view(-1, batch_size, self.vocab_size)[-1]  # .permute(1, 0, 2)

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if cfg.device:
            samples = samples.to(cfg.device)

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(1, -1)

            out, hidden = self.gen.forward(inp, hidden)

        return samples
