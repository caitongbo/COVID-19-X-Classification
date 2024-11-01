from torch import nn
from torchvision.models import resnet50, resnet101
from torchvision.models import mobilenet_v2
import timm
import torch

from .resnet_csra import ResNet_CSRA
# from .resnetcsra import ResNet_CSRA
from .resnet_csra_fpn import ResNet_CSRA_FPN

from .timm.models.vision_transformer import vit_small_patch16_224
# from .timm.models.vision_transformer_MCRA import vit_base_patch16_224

from .timm.models.swin_transformer import swin_tiny_patch4_window7_224
# from .timm.models.swin_transformer_MCRA import swin_tiny_patch4_window7_224
from .cswin import CSWin_64_12211_tiny_224
from .timm.models.tnt import tnt_s_patch16_224
from  .pvt import pvt_v2_b2
from .mobilev2_ca import mbv2_ca
from .coatnet import coatnet_1
# from .vit_fpn import vit_base_patch16_224

from .cvt.cvt import CvT
from .covidnet import CovidNet



class CSRA(nn.Module): # one basic block 
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda                        
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        # score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
        score = self.head(x)

        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99: # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        # return 0.3 * base_logit + 0.7*att_logit
        return att_logit
        # return self.lam * base_logit + att_logit



class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit



class COVIDNext50(nn.Module):
    def __init__(self, net, n_classes):
        super(COVIDNext50, self).__init__()

        self.classifier = MHA(1, 1, 256, 3) 
        self.n_classes = n_classes
        self.isMCRA = False

        if net=='resnet50-nocutmix':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Linear(2048,3)
            self.net =self.resnet
        if net=='resnet50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Linear(2048,3)
            self.net =self.resnet
        if net=='resnet101':
            self.resnet = resnet101(pretrained=True)
            self.resnet.fc = nn.Linear(2048,3)
            self.net =self.resnet
        if net == 'resnet_fpn':
            from model.pytorch_fpn.fpn.factory import make_fpn_resnet
            self.model = make_fpn_resnet(
                name='resnet50',
                fpn_type='panet',
                pretrained=True,
                num_classes=3,
                fpn_channels=256,
                in_channels=3,
                out_size=(224, 224))
                # Classifier head

            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.conv_cls_head = nn.Linear(int(3), 3)
            self.net = self.model

        if net == 'resnet_MCRA':
            from .test_t1 import resnet50_fpn_backbone
            self.model = resnet50_fpn_backbone(pretrain_path="/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
                                            norm_layer=torch.nn.BatchNorm2d,
                                            trainable_layers=3)
                
            self.net = self.model
            self.isMCRA = True

        if net == 'resnet_csra':
            self.resnet_csra = ResNet_CSRA(num_heads=1, lam=0.1, depth=50, num_classes=3, cutmix=None)
            self.net =self.resnet_csra
        if net == 'resnet_csra_fpn':
            self.resnet_csra = ResNet_CSRA_FPN(num_heads=1, lam=0.1, depth=50, num_classes=3, cutmix=None)
            self.net =self.resnet_csra
        if net == 'vit':
            self.vit =  timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=3)
            self.net =self.vit
        if net == 'vit_csra':
            self.vit = vit_small_patch16_224(pretrained=True, num_classes=3)
            self.net =self.vit
        if net == 'vit_MCRA':
            self.vit = vit_small_patch16_224(pretrained=False, num_classes=3)
            self.net =self.vit
        if net == 'swin':
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=3)
            self.net =self.model
        if net == 'swin_csra':
            self.model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=3)  
            self.net =self.model
        if net == 'swin_MCRA':
            self.model = swin_tiny_patch4_window7_224(pretrained=False, num_classes=3)
            self.net = self.model
        if net =='mobilenet_v2':
            self.mobilenet_v2 = mbv2_ca()
            self.net = self.mobilenet_v2
        if net == 'coatnet':
            self.net = coatnet_1()
        if net == 'cvt':
            self.net = CvT(224, 3, 3)
        if net == 'cswin':
            self.net = CSWin_64_12211_tiny_224(pretrained=True, num_classes=3)
        if net == 'volo':
            self.net = timm.create_model('volo_d1_224', pretrained=True, num_classes=3)
        if net == 'poolformer':
            self.net = timm.create_model('poolformer_s36', pretrained=True, num_classes=3)
        if net == 'convmixer':
            self.net = timm.create_model('convmixer_768_32', pretrained=True, num_classes=3)
        if net == 'convnext':
            self.net = timm.create_model('convnext_tiny', pretrained=True, num_classes=3)
        if net == 'inception_v3':
            self.net = timm.create_model('inception_v3', pretrained=True, num_classes=3)
        if net == 'resnest50d':
            self.net = timm.create_model('resnest50d', pretrained=True, num_classes=3)
        if net == 'covidnet':
            self.net = CovidNet('large', n_classes=3)
        if net == 'vgg19':
            self.net = timm.create_model('vgg19', pretrained=True, num_classes=3)
        if net == 'deit_s':
            self.net = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=3)
        if net == 'efficientnet':
            self.net = timm.create_model('efficientnet_b5', pretrained=True, num_classes=3)
        if net == 'regnety_8gf':
            self.net = timm.create_model('regnety_080', pretrained=True, num_classes=3)
        if net == 'mobilenetv3':
            self.net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=3)           
        if net == 'densenet':
            self.net = timm.create_model('densenet121', pretrained=True, num_classes=3)   
        if net == 'tnt':
            self.net = timm.create_model('tnt_s_patch16_224', pretrained=True, num_classes=3)     
        if net == 'xception':
            self.net = timm.create_model('xception', pretrained=True, num_classes=3)      
        if net == 'cspresnet':
            self.net = timm.create_model('cspresnet50', pretrained=True, num_classes=3)               
        self.head = nn.Conv2d(256, 3, 1, bias=False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256), 3)

    def forward(self, input):

        x= self.net(input)
        if self.isMCRA:
            x = self.classifier(x)   # MCRA for ResNet 50


        # x = self.pooling(x).flatten(1)
        # x = self.conv_cls_head(x)
        return x

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)