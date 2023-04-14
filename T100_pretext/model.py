import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn.functional as func
import torchvision

class PretextNet(nn.Module):
    def __init__(self, args, CNN_model_path=None):
        super(PretextNet, self).__init__()
        
        self.backbone_dim = args.image_dim
        if CNN_model_path:
            self.vgg = torchvision.models.vgg16()
            state_dict = torch.load(CNN_model_path)
            self.vgg.load_state_dict(state_dict)
        else:
            self.vgg = torchvision.models.vgg16(pretrained=True)

        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:5])
        # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        '''
        print(self.vgg.classifier)
            Sequential(
              (0): Linear(in_features=25088, out_features=4096, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.5, inplace=False)
              (3): Linear(in_features=4096, out_features=4096, bias=True)
              (4): ReLU(inplace=True)
              (5): Dropout(p=0.5, inplace=False)
              (6): Linear(in_features=4096, out_features=1000, bias=True)
            )
        '''
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        # feature:
        # fc7 of vgg16
        feat = self.vgg.features(x)
        feat = feat.view(feat.shape[0], -1)
        feat = self.vgg.classifier(feat)
        # contrastive head:
        # 4096 --> 512
        output = self.contrastive_head(feat)

        return feat, output

