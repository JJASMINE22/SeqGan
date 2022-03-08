# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import json
import torch
import numpy as np
import config as cfg
from torch import nn
from seqgan import SeqGAN
from utils.generate import Generator


if __name__ == '__main__':

    seq_gan = SeqGAN(gen_embed_dim=cfg.gen_embed_dim,
                     gen_hidden_dim=cfg.gen_hidden_dim,
                     vocab_size=cfg.vocab_size,
                     max_seq_len=cfg.max_seq_len,
                     dis_embed_dim=cfg.dis_embed_dim,
                     filter_sizes=cfg.filter_sizes,
                     num_filters=cfg.num_filters,
                     Lambda=cfg.Lambda,
                     gen_lr=cfg.gen_lr,
                     dis_lr=cfg.dis_lr,
                     batch_size=cfg.batch_size,
                     rollout_num=cfg.rollout_num,
                     dropout=cfg.dropout,
                     padding_idx=cfg.padding_idx,
                     device=cfg.device,
                     ignore_pretrain=cfg.ignore_pretrain,
                     gen_ckpt_path=cfg.pre_gen_checkpoint_path + "\\Epoch080_pre_gen_loss3.083.pth.tar",
                     dis_ckpt_path=cfg.pre_dis_checkpoint_path + "\\Epoch015_pre_dis_loss3.71078.pth.tar")

    data_gen = Generator(file_path=cfg.datapath,
                         vocab_path=cfg.vocabpath,
                         vocab_size=cfg.vocab_size,
                         max_seq_len=cfg.max_seq_len,
                         batch_size=cfg.batch_size,
                         train_ratio=cfg.train_ratio,
                         dataset=cfg.dataset)

    with open(cfg.vocabpath + '\\{}.json'.format(cfg.dataset), 'r') as f:
        dict = json.load(f)
    vocab = dict['vocab']

    def decode(samples):
        decode_mat = []
        for i, sample in enumerate(samples.numpy()):
            token = ''
            for idx in sample:
                token += np.array(vocab)[idx] + ' '
            decode_mat.append(token.strip())
        decode_mat = np.array(decode_mat).reshape([-1, 1])

        return decode_mat

    if not cfg.ignore_pretrain:
        print('Starting Pretraining...')
        # === pretrain ===
        # for generator
        for epoch in range(cfg.MLE_train_epoch):
            g_train_gen = data_gen.generate(training=True)
            for iter in range(data_gen.get_train_len()):
                real_sources = next(g_train_gen)
                seq_gan.pretrain_generator(real_sources)

            print('pretrain gen loss is {:.3f}'.format(seq_gan.pre_gen_loss / data_gen.get_train_len()))

            torch.save({'state_dict': seq_gan.gen.state_dict(),
                        'pre_gen_loss': seq_gan.pre_gen_loss / data_gen.get_train_len()},
                       cfg.pre_gen_checkpoint_path + '\\Epoch{:0>3d}_pre_gen_loss{:.3f}.pth.tar'.format(
                           epoch + 1,
                           seq_gan.pre_gen_loss / data_gen.get_train_len()
                       ))

            samples = seq_gan.gen.sample(num_samples=cfg.batch_size,
                                         batch_size=cfg.batch_size)
            print('generate samples: ', decode(samples))
            seq_gan.pre_gen_loss = 0

        # for discriminator
        for step in range(cfg.d_step):
            d_train_gen = data_gen.generate(training=True)
            for epoch in range(cfg.d_epoch):
                for iter in range(data_gen.get_train_len()):
                    real_sources = next(d_train_gen)
                    fake_sources = seq_gan.gen.sample(num_samples=real_sources.shape[0],
                                                      batch_size=real_sources.shape[0])
                    seq_gan.train_discriminator(real_sources, fake_sources)

                print('pretrain dis loss is {:.5f}'.format(seq_gan.dis_loss / data_gen.get_train_len()),
                      'pretrain dis acc is {:.3f}'.format(seq_gan.dis_acc / data_gen.get_train_len() * 100))

                torch.save({'state_dict': seq_gan.dis.state_dict(),
                            'loss': seq_gan.dis_loss / data_gen.get_train_len()},
                           cfg.pre_dis_checkpoint_path + '\\Epoch{:0>3d}_pre_dis_loss{:.5f}.pth.tar'.format(
                               step * cfg.d_epoch + epoch + 1,
                               seq_gan.dis_loss / data_gen.get_train_len()
                           ))
                seq_gan.dis_loss, seq_gan.dis_acc = 0, 0

    print('Starting Adversarial Training...')
    # === ADV Train ===
    for adv_epoch in range(cfg.ADV_train_epoch):
        # for generator
        for step in range(cfg.ADV_g_step):
            seq_gan.adv_train_generator()

        print('adv train gen loss is {:.3f}'.format(seq_gan.adv_gen_loss/cfg.ADV_g_step))

        torch.save({'state_dict': seq_gan.gen.state_dict(),
                    'adv_gen_loss': seq_gan.adv_gen_loss/cfg.ADV_g_step},
                   cfg.adv_gen_checkpoint_path + '\\Epoch{:0>3d}_adv_gen_loss{:.3f}.pth.tar'.format(
                       adv_epoch*cfg.ADV_g_step + step + 1,
                       seq_gan.adv_gen_loss/cfg.ADV_g_step
                   ))

        samples = seq_gan.gen.sample(num_samples=cfg.batch_size,
                                     batch_size=cfg.batch_size)
        print('generate samples: ', decode(samples))
        seq_gan.adv_gen_loss = 0

        # for discriminator
        for step in range(cfg.d_step):
            d_train_gen = data_gen.generate(training=True)
            for epoch in range(cfg.d_epoch):
                for iter in range(data_gen.get_train_len()):
                    real_sources = next(d_train_gen)
                    fake_sources = seq_gan.gen.sample(num_samples=real_sources.shape[0],
                                                      batch_size=real_sources.shape[0])
                    seq_gan.train_discriminator(real_sources, fake_sources)

                print('adv dis loss is {:.5f}'.format(seq_gan.dis_loss/data_gen.get_train_len()),
                      'adv dis acc is {:.3f}'.format(seq_gan.dis_acc/data_gen.get_train_len()*100))

                torch.save({'state_dict': seq_gan.dis.state_dict(),
                            'loss': seq_gan.dis_loss/data_gen.get_train_len()},
                           cfg.adv_dis_checkpoint_path + '\\Epoch{:0>3d}_adv_dis_loss{:.5f}.pth.tar'.format(
                               adv_epoch*cfg.d_step*cfg.d_epoch+step*cfg.d_epoch+epoch+1,
                               seq_gan.dis_loss/data_gen.get_train_len()
                           ))

                seq_gan.dis_loss, seq_gan.dis_acc = 0, 0
