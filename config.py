import torch

# ===basic===
vocab_size = 4500
max_seq_len = 20
inter_epoch = 15
ADV_train_epoch = 2000
batch_size = 64
train_ratio = 0.7
Lambda = 1e-3
padding_idx = None
ignore_pretrain = True
device = torch.device('cuda') if torch.cuda.is_available() else None

# ===generator===
no_log = False
gen_embed_dim = 32
gen_hidden_dim = 32
gen_init = 'normal'
gen_lr = 1e-2
MLE_train_epoch = 80
ADV_g_step = 1
rollout_num = 16
pre_gen_checkpoint_path = '.\\saved\\pre_generator'
adv_gen_checkpoint_path = '.\\saved\\adv_generator'

# ===discriminator===
dis_embed_dim = 64
filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_init = 'normal'
dropout = 0.25
loss_mode = 'Hinge'
dis_lr = 1e-4
d_step = 5
d_epoch = 3
ADV_d_step = 5
ADV_d_epoch = 3
pre_dis_checkpoint_path = '.\\saved\\pre_discriminator'
adv_dis_checkpoint_path = '.\\saved\\adv_discriminator'

# ===train===
datapath = 'C:\\DATASET\\nature_language'
dataset = 'image_coco'
vocabpath = 'C:\\PythonProjects\\SeqGan\\vocab'
start_letter = 1
