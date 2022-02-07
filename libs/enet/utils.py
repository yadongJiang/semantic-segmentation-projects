import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Conv2d(in_chans, out_chans - 3, 3, 2, 1, bias=False)
        self.ext_branch = nn.MaxPool2d(3, 2, 1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        
        out = torch.cat([main, ext], dim=1)
        out = self.relu(self.bn(out))
        
        return out

class DownsamplingBottleneck(nn.Module):
    def __init__(self, 
                 in_chans, 
                 out_chans, 
                 internal_ratio=4, 
                 return_indices=False, 
                 dropout_prob=0, ):
        super(DownsamplingBottleneck, self).__init__()
        self.return_indices = return_indices
        
        internal_channels = in_chans // internal_ratio
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_chans, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_chans, 1, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

        self.ext_regul = nn.Dropout(p=dropout_prob)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        bs, ext_chans, h, w = ext.size()
        padding = torch.zeros(bs, ext_chans - main.size()[1], h, w)

        if main.is_cuda:
            padding = padding.cuda()
        
        main = torch.cat([main, padding], dim=1)
        out = main + ext
        return self.act(out), max_indices

class RegularBottleneck(nn.Module):
    def __init__(self, 
                 in_channs, 
                 internal_ratio=4, 
                 kernel_size=3, 
                 padding=0, 
                 dilation=1, 
                 asymmetric=False, 
                 dropout_prob=0):
        super(RegularBottleneck, self).__init__()
        internal_channels = in_channs // internal_ratio

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channs, internal_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(internal_channels), 
            nn.ReLU(inplace=True)
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, 
                          internal_channels, 
                          kernel_size=(kernel_size, 1), 
                          stride=1, 
                          padding=(padding, 0), 
                          dilation=dilation, 
                          bias=False),
                nn.BatchNorm2d(internal_channels), 
                nn.ReLU(inplace=True),

                nn.Conv2d(internal_channels, 
                          internal_channels, 
                          kernel_size=(1, kernel_size), 
                          stride=1, 
                          padding=(0, padding), 
                          dilation=dilation, 
                          bias=False),
                nn.BatchNorm2d(internal_channels), 
                nn.ReLU(inplace=True)
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, 
                          internal_channels, 
                          kernel_size=kernel_size, 
                          stride=1, 
                          padding=padding, 
                          dilation=dilation, 
                          bias=False),
                nn.BatchNorm2d(internal_channels), 
                nn.ReLU(inplace=True)
            )
        
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, in_channs, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(in_channs), 
            nn.ReLU(inplace=True)
        )
        self.ext_regul = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        
        out = main + ext
        return self.relu(out)

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_chans, out_chans, internal_ratio=4, dropout_prob=0):
        super(UpsamplingBottleneck, self).__init__()
        internal_channels = in_chans // internal_ratio
        
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chans)
        )

        self.main_uppool1 = nn.MaxUnpool2d(kernel_size=2)
        
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_chans, internal_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(internal_channels)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(internal_channels, internal_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(internal_channels), 
            nn.ReLU(inplace=True)
        )

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_chans, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chans)
        )
        self.ext_regul = nn.Dropout(p =dropout_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_uppool1(main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.upsample(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        out = main + ext
        return self.relu(out)