from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .csra import CSRA, MHA


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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
        # print(x.shape)
        score = score.flatten(2)
        # print(score)
        # print(score.shape)

        base_logit = torch.mean(score, dim=2)

        if self.T == 99: # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
                # print(score_soft)
            att_logit = torch.sum(score * score_soft, dim=2)

        # return base_logit + self.lam * att_logit
        return att_logit
        # return base_logit + att_logit



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
        
class ResNet_CSRA_FPN(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=50, input_dim=2048, cutmix=None,scale=1):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA_FPN, self).__init__(self.block, self.layers)

        self.init_weights(pretrained=True, cutmix=cutmix)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(256)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(256)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(256)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(256)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

        # self.coordatt = CoordAtt(3,3)

        self.scale = scale
        # self.fc = nn.Conv2d(in_channels=2048, out_channels=3, kernel_size=1, stride=1, bias=False)

        # self.loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05,disable_torch_grad_focal_loss=True)

        self.loss_func = F.binary_cross_entropy_with_logits
        self.classifier = MHA(num_heads, lam, input_dim=3, num_classes=3)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear',align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear',align_corners=True) + y

    def backbone(self, x):
        h = x

        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)


        h = self.layer1(h)

        c2 = h
        h = self.layer2(h)

        c3 = h
        h = self.layer3(h)

        c4 = h
        h = self.layer4(h)
        c5 = h

        # Top-down
        p5 = self.toplayer(c5)

        p5 = self.toplayer_relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)

        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))


        p4 = self._upsample_add(p5, c4)

        p4 = self.smooth1(p4)

        p4 = self.smooth1_relu(self.smooth1_bn(p4))
        # print(p4.shape)

        c3 = self.latlayer2(c3)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)

        # out = self.coordatt(out)

        out = self._upsample(out, x, scale=self.scale)
        # print(out.size())

        # out = out.view(out.size(0), -1)         
        # out = self.fc(out)
        return out

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        # loss = self.loss_func(logit, target)
        loss = self.loss_func(logit, target, reduction="mean")

        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        if pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            pretrained_state_dict = model_zoo.load_url(model_url)
            now_state_dict = self.state_dict()
            now_state_dict.update(pretrained_state_dict)
            self.load_state_dict(now_state_dict)
        # elif pretrained:
        #     print("backbone params inited by Pytorch official model")
        #     model_url = model_urls["resnet{}".format(self.depth)]
        #     state_dict = model_zoo.load_url(model_url)
        # model_dict = self.state_dict()
        # try:
        #     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        #     self.load_state_dict(pretrained_dict)
        # except:
        #     logger = logging.getLogger()
        #     logger.info(
        #         "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
        #     state_dict = self._keysFix(model_dict, state_dict)
        #     self.load_state_dict(state_dict)
        # self.fc = nn.Linear(2048,3)

        # remove the original 1000-class fc
        self.fc = nn.Sequential() 