import math

import torch
import torch.nn as nn

import config as cfg


class LSTMGenerator(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 max_seq_len: int,
                 padding_idx: int,
                 **kwargs):
        super(LSTMGenerator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size,
                                       embedding_dim=self.embed_dim,
                                       padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim)
        self.lstm2out = nn.Linear(in_features=self.hidden_dim,
                                  out_features=self.vocab_size)

        self.weights = self.init_params()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inp, hidden):
        """
        Forbidden batch first
        Can be used not only for single participle reasoning, but also for multiple participles
        :param inp: seq_len * batch_size
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """

        assert len(inp.size()) > 1

        emb = self.embeddings(inp)  # seq_len * batch_size * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: seq_len * batch_size * hidden_dim
        out = out.view(-1, self.hidden_dim)  # out: (seq_len * batch_size) * hidden_dim
        out = self.lstm2out(out)  # (seq_len * batch_size) * vocab_size

        if cfg.no_log:
            pred = torch.softmax(out, dim=-1)
        else:
            pred = torch.log_softmax(out, dim=-1)

        return pred, hidden

    def forward_seqgan(self, sentences, start_letter=cfg.start_letter, if_sample=False):
        """
        This method is used for circular reasoning with single participles
        :param sentences: original sentences, started with '/s'
        :param start_letter: '/s'
        :param if_sample: Decide whether to infer samples with '/s', not to use origin sentences
        """
        batch_size = sentences.size()[0]

        inp = torch.LongTensor([start_letter, ]*batch_size).view([1, -1])
        samples = torch.zeros((batch_size, self.max_seq_len))
        seq_out_array = torch.zeros((batch_size, self.max_seq_len, self.vocab_size))

        if cfg.device:
            inp = inp.to(cfg.device)
            sentences = sentences.to(cfg.device)
            seq_out_array = seq_out_array.to(cfg.device)

        hidden_state, cell_state = self.init_hidden(batch_size=batch_size)
        for i in range(self.max_seq_len):

            if not if_sample:
                if i:
                    inp = sentences[:, i-1].view([1, -1])

            out, [hidden_state, cell_state] = self.forward(inp, [hidden_state, cell_state])
            seq_out_array[:, i, :] = out

            if not cfg.no_log:
                out = torch.exp(out)
            out = torch.multinomial(out, 1).view(-1)  # [batch_size] (sampling from each row)
            samples[:, i] = out.data
            inp = out.view([1, -1])

        return samples, seq_out_array, [hidden_state, cell_state]

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        fake_sentences = torch.zeros(size=(batch_size, self.max_seq_len))

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            out, _, _ = self.forward_seqgan(fake_sentences, if_sample=True)
            samples[b * batch_size:(b + 1) * batch_size, :] = out
        samples = samples[:num_samples]

        return samples

    def init_params(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] is 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1 / math.sqrt(param.shape[0])
                    if cfg.dis_init == 'uniform':
                        torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                        weights.append(param)
                    elif cfg.dis_init == 'normal':
                        torch.nn.init.normal_(param, std=stddev)
                        weights.append(param)

        return weights

    def init_hidden(self, batch_size=cfg.batch_size):
        # set batch_size to 1, and use broadcast
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if cfg.device:
            return h.to(cfg.device), c.to(cfg.device)
        else:
            return h, c
