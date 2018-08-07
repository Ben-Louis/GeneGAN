import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *


class Encoder(Module):
    def __init__(self, conv_dim, out_dim):
        super(Encoder, self).__init__()

        layers = []
        layers.append(ConvBlock(3, conv_dim, norm='in', act='relu', kernel_size=7, stride=1, padding=3))
        layers.append(ConvBlock(conv_dim, conv_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=2, dilation=2))
        layers.append(ConvBlock(conv_dim, conv_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1))
        # (B, conv_dim*2, imsize//2, imsize//2)

        curr_dim = conv_dim * 2
        layers.append(ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=2, dilation=2))
        layers.append(ResBlock(curr_dim, dilation=2))        
        layers.append(ResBlock(curr_dim, dilation=2)) 
        layers.append(ConvBlock(curr_dim, curr_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1))
        # (B, conv_dim*2, imsize//4, imsize//4)

        curr_dim = curr_dim * 2
        layers.append(ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        layers.append(ResBlock(curr_dim, dilation=2))        
        layers.append(ResBlock(curr_dim, dilation=2)) 
        layers.append(ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))        
        # (B, conv_dim*2, imsize//4, imsize//4)
        self.in_conv = Sequential(*layers)

        ##### holatile branch #####
        self.hol_conv = Sequential(
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, out_dim[0], norm='in', act='relu', kernel_size=3, stride=1, padding=1)
            )

        ##### attr branches #####
        # now we assume len(out_dim)=2
        self.stn = STN(curr_dim)
        self.attr_conv = Sequential(
            ConvBlock(curr_dim, curr_dim//2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim//2, dilation=2),
            ResBlock(curr_dim//2, dilation=2),
            ResBlock(curr_dim//2, dilation=2),
            ConvBlock(curr_dim//2, out_dim[1], norm='in', act='relu', kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        x = self.in_conv(x)
        out_hol = self.hol_conv(x)
        feat, inv_theta = self.stn(x)
        out_attr = self.attr_conv(feat)
        inv_theta = inv_theta.view(x.size(0), 6).unsqueeze(2).unsqueeze(3).expand(x.size(0), 6,out_attr.size(2),out_attr.size(3))
        out_attr = torch.cat([out_attr, inv_theta], dim=1)
        return [out_hol, out_attr]

class Decoder(Module):
    def __init__(self, conv_dim, out_dim):
        super(Decoder, self).__init__()

        curr_dim = conv_dim * 4
        self.hol_conv = Sequential(
            ConvBlock(out_dim[0], curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1),
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            nn.ReLU(True)
            )

        self.attr_conv = Sequential(
            ConvBlock(out_dim[1], curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1),
            ResBlock(curr_dim, dilation=2),
            ResBlock(curr_dim, dilation=2),
            ResBlock(curr_dim, dilation=2),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1)                        
            )

        layers = []

        layers.append(ConvBlock(curr_dim*2, curr_dim, norm='in', act='relu', kernel_size=1, stride=1, padding=0))
        layers.append(Self_Attn(curr_dim))
        layers.append(nn.ReLU())
        layers.append(ResBlock(curr_dim, dilation=2))
        layers.append(Self_Attn(curr_dim))
        layers.append(nn.ReLU())
        layers.append(ResBlock(curr_dim, dilation=2))  
        layers.append(ConvBlock(curr_dim, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1))      
        # (B, conv_dim*2, imsize/2, imsize/2)
        curr_dim = curr_dim // 2

        layers.append(ResBlock(curr_dim, dilation=2))
        layers.append(ResBlock(curr_dim, dilation=2))
        layers.append(ResBlock(curr_dim, dilation=2))
        layers.append(ConvBlock(curr_dim, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1))
        curr_dim = curr_dim // 2

        layers.append(ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=2, dilation=2))
        layers.append(ConvBlock(curr_dim, 3, norm='', act='tanh', kernel_size=7, stride=1, padding=3))

        self.out_conv = Sequential(*layers)


    def forward(self, feats):
        in_hol, in_attr = feats
        in_attr, stn_attr = in_attr[:,:-6,:,:], in_attr[:,-6:,:,:]
        stn_attr = stn_attr[:,:,0,0].view(-1,2,3)

        feat_hol = self.hol_conv(in_hol)
        feat_attr = self.attr_conv(in_attr)
        feat_attr = STN.affine_map(feat_attr, stn_attr)

        feat = torch.cat([feat_hol, feat_attr], dim=1)
        return self.out_conv(feat)


class Discriminator(Module):
    def __init__(self, conv_dim, image_size):
        super(Discriminator, self).__init__()

        layers = []

        layers.append(ConvBlock(3, conv_dim, norm='', act='lrelu', sn=True, kernel_size=5, stride=1, padding=2))
        curr_dim = conv_dim

        while image_size > 4:
            layers.append(ConvBlock(curr_dim, curr_dim*2, norm='', act='lrelu', sn=True, kernel_size=4, stride=2, padding=1))
            curr_dim = curr_dim * 2
            image_size = image_size // 2

        layers.append(ConvBlock(curr_dim, 1, norm='', act='', sn=False, kernel_size=3, stride=1, padding=0))
        self.main = Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(x.size(0),-1).mean(dim=1)







