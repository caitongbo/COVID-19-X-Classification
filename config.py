name ="swin_base_path4_csra"

gpu = True
batch_size = 64
val_batch_size = 128

n_threads = 16
random_seed = 1337
cutmix = True

models = ['resnet','resnet_csra','resnet_csra_fpn','vit','vit_csra','swin','swin_csra']

# Model weights path  
########covidx
# weights = ["/home/ubuntu/caitongbo/COVID-Net-Pytorch/models/ckpts4/resnet50_cutmix/resnet50_cutmix_Acc_95.00_F1_94.18_pre_93.62_rec_94.83_step_6400.pth",
#         "./models/ckpts4/resnet50_csra_cutmix9_no_norm/resnet50_csra_cutmix9_no_norm_Acc_96.50_F1_95.82_pre_95.77_rec_96.17_step_16800.pth",
#         "./models/ckpts4/resnet50_fpn_csra_cutmix9_no_norm/resnet50_fpn_csra_cutmix9_no_norm_Acc_97.50_F1_97.03_pre_96.82_rec_97.33_step_24000.pth",
#         "./models/ckpts4/vit_base_16/vit_base_16_Acc_96.75_F1_96.18_pre_96.04_rec_96.33_step_8400.pth",
#         "./models/ckpts4/vit_csra_base_16/vit_csra_base_16_Acc_97.25_F1_96.70_pre_96.62_rec_97.00_step_16400.pth",
#         "./models/ckpts/vit_MCRA/vit_MCRA_Acc_97.25_F1_96.72_pre_96.70_rec_97.00_step_18400.pth",
#         "./models/ckpts4/swin_base_path4/swin_base_path4_Acc_96.50_F1_95.90_pre_95.61_rec_96.33_step_19200.pth",
#         "./models/ckpts4/swin_base_path4_csra/swin_base_path4_csra_Acc_96.75_F1_96.32_pre_96.12_rec_96.67_step_8400.pth"]

#########covidx 6:2:2
weights = ["./models/ckpts/resnet50_nocutmix/resnet50_Acc_96.56_F1_95.14_pre_95.02_rec_95.27_step_6000.pth",
"./models/ckpts/resnet50/resnet50_Acc_96.56_F1_95.13_pre_95.32_rec_95.02_step_4000.pth",
"./models/ckpts/resnet_csra/resnet_csra_Acc_97.20_F1_95.96_pre_96.09_rec_95.86_step_16800.pth",
"./models/ckpts2/vit/vit_Acc_96.86_F1_95.47_pre_95.55_rec_95.42_step_24800.pth",
"./models/ckpts/vit_csra/vit_csra_Acc_96.71_F1_95.23_pre_95.45_rec_95.09_step_28000.pth",
"./models/ckpts/swin/swin_Acc_97.02_F1_95.71_pre_95.87_rec_95.59_step_8000.pth",
"./models/ckpts/swin_csra/swin_csra_Acc_96.99_F1_95.70_pre_95.76_rec_95.66_step_5600.pth",
"./models/ckpts/cswin/cswin_Acc_94.35_F1_92.10_pre_92.60_rec_91.82_step_28800.pth",
"./models/ckpts/convnext/convnext_Acc_97.08_F1_95.76_pre_96.10_rec_95.52_step_14000.pth",
"./models/ckpts/convmixer/convmixer_Acc_97.28_F1_96.05_pre_96.33_rec_95.84_step_23600.pth",
"./models/ckpts/poolformer/poolformer_Acc_96.94_F1_95.55_pre_95.82_rec_95.37_step_24800.pth",
"./models/ckpts/volo/volo_Acc_97.25_F1_95.95_pre_95.98_rec_95.94_step_11600.pth",
"./models/ckpts/inception_v3/inception_v3_Acc_96.56_F1_95.07_pre_95.56_rec_94.76_step_6800.pth",
"./models/ckpts/resnest50d/resnest50d_Acc_97.08_F1_95.77_pre_95.81_rec_95.73_step_7600.pth",
"./models/ckpts/covidnet/covidnet_Acc_94.71_F1_92.54_pre_92.88_rec_92.30_step_27200.pth",
"./models/ckpts/resnet_MCRA/resnet_MCRA_Acc_95.86_F1_94.33_pre_94.29_rec_94.39_step_1600.pth",
"./models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.05_F1_95.69_pre_95.93_rec_95.52_step_19200.pth",
"./models/ckpts/vgg19/vgg19_Acc_96.23_F1_94.75_pre_95.05_rec_94.63_step_3600.pth",
"./models/ckpts/mobilenetv3/mobilenetv3_Acc_96.18_F1_94.47_pre_94.74_rec_94.26_step_9600.pth",
"./models/ckpts/efficientnet/efficientnet_Acc_93.29_F1_90.88_pre_90.99_rec_90.91_step_24000.pth",
"./models/ckpts/densenet/densenet_Acc_96.33_F1_94.81_pre_94.90_rec_94.81_step_4000.pth",
"./models/ckpts/deit_s/deit_s_Acc_96.71_F1_95.22_pre_95.24_rec_95.21_step_18800.pth",
"./models/ckpts/regnety_8gf/regnety_8gf_Acc_97.07_F1_95.74_pre_96.14_rec_95.45_step_17200.pth",
"./models/ckpts/tnt/tnt_Acc_97.07_F1_95.72_pre_96.36_rec_95.28_step_4800.pth",
"./models/ckpts/xception/xception_Acc_96.17_F1_94.59_pre_94.52_rec_94.69_step_23200.pth",
"./models/ckpts/cspresnet/cspresnet_Acc_96.48_F1_94.89_pre_95.29_rec_94.61_step_8000.pth",
"./models/ckpts/resnet_MCRA_best/resnet_MCRA_Acc_97.20_F1_95.92_pre_96.29_rec_95.66_step_16400.pth"]


# Optimizer
lr = 1e-4
weight_decay = 1e-3
lr_reduce_factor = 0.7
lr_reduce_patience = 5

# Data
# train_imgs = "./dataset/covidx3/data/train"
# train_labels = "./dataset/covidx3/data/train_split_v3.txt"

# val_imgs = "./dataset/covidx3/data/test"
# val_labels = "./dataset/covidx3/data/test_split_v3.txt"

train_imgs = "./dataset/covidx9/train"
train_labels = "/data/our_train.txt"

val_imgs = "./dataset/covidx9/test"
val_labels = "/data/our_val.txt"

test_labels = "/data/our_test.txt"

#
sample = './test/'
# Categories mapping
mapping = {
    'normal': 0,
    'pneumonia': 1,
    'COVID-19': 2
}
# Loss weigths order follows the order in the category mapping dict
loss_weights = [0.05, 0.05, 1.0]
# loss_weights = [1, 1.46, 16.8]


width = 224
height = 224

# width = 480
# height = 480

n_classes = len(mapping)

# Training
epochs = 150
log_steps = 200
eval_steps = 400
ckpts_dir = "./models/ckpts"
