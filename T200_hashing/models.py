import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn.functional as func
import torchvision

class PretextNet(nn.Module):
    def __init__(self, args):
        super(PretextNet, self).__init__()
        # CNN_model_path = '/home/zl/.cache/torch/checkpoints/vgg16-397923af.pth'
        CNN_model_path = False
        self.backbone_dim = 4096
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
        # todo
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

class FuseTransEncoder(nn.Module):
    def __init__(self, args):
        super(FuseTransEncoder, self).__init__()
        self.image_dim = args.image_dim  # 4096
        self.common_dim = args.common_dim # 256
        self.topk = args.topk
        self.batchsize = args.batchsize
        self.nbit = args.nbit
        self.alpha = args.alpha

        self.nhead = args.nhead
        self.act = args.trans_act
        self.dropout = args.dropout
        self.num_layer = args.num_layer

        ''' MLP '''
        self.imageMLP = nn.Sequential(
            nn.Linear(self.image_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.Tanh()
        )

        ''' TEs encoder '''
        self.imagePosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)

        imageEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.common_dim,
                                                    activation=self.act,
                                                    dropout=self.dropout)
        imageEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer,
                                                          num_layers=self.num_layer,
                                                          norm=imageEncoderNorm)

        ''' hash'''
        self.hash = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.BatchNorm1d(self.nbit),
            nn.Tanh(),
        )

        ''' decoder'''
        self.imageDecoder = nn.Sequential(
            nn.Linear(self.nbit, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),

            nn.Linear(1024, self.image_dim),
            nn.BatchNorm1d(self.image_dim),
            nn.Tanh(),
        )

    def forward(self, img, nbrs):
        nsample = img.size(0)

        # (1) MLP
        ''' image '''
        img_mlp = self.imageMLP(img)
        img_mlp = img_mlp.unsqueeze(1)  # [n, 1, 256]

        ''' neigboors '''
        nbrs_mlp = torch.zeros((nsample, self.topk, self.common_dim)).cuda()
        for i in range (self.topk):
            # (1) Get neighbor
            nbr = nbrs[:, i, :]

            # (2) MLP
            nbr_mlp = self.imageMLP(nbr) # [n, 256]
            nbrs_mlp[:, i, :] = nbr_mlp

        # (2) construct sequence
        img_nbrs = torch.cat((img_mlp, nbrs_mlp), dim=1) # [n, 1+k, com_dim]

        # (3) PosEncoder
        tempPos = img_nbrs.permute(1, 0, 2)  # [s, n, d]
        imageSrc = self.imagePosEncoder(tempPos)

        # (4) TransformerEncoder
        imageMemory = self.imageTransformerEncoder(imageSrc)
        tempTes = imageMemory.permute(1, 0, 2) # [n, s, d]
        img_tes = tempTes[:, 0, :]
        img_tes_nei = torch.mean(tempTes[:, 1:, :], dim=1)
        img_tes_fuse = self.alpha * img_tes + (1-self.alpha) * img_tes_nei

        # (5) Hash
        hashcode = self.hash(img_tes_fuse)

        # (6) Decoder
        img_reconst = self.imageDecoder(hashcode)

        return hashcode, img_reconst

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    Refer:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

