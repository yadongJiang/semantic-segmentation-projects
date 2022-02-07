import torch
import torch.nn as nn

class AddCoords(nn.Module):
    
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size() # b, c, h, w

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1) # (1, w, h) (1, y_dim, x_dim)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2) # (1, h, w) ==> (1, w, h) (1, y_dim, x_dim)

        xx_channel = xx_channel.float() / (x_dim - 1) # 归一化到0~1
        yy_channel = yy_channel.float() / (y_dim - 1) # 归一化到0~1

        xx_channel = xx_channel * 2 - 1 # 归一化到-1. ~ 1.
        yy_channel = yy_channel * 2 - 1 # 归一化到-1. ~ 1.

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3) # [batch_size, 1, h, w], 代表坐标值y(高)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3) # [batch_size, 1, h, w], 代表坐标值x(宽)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1) # [batch_size, c+2, h, w]

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class Conv2D_Norm_Activation(nn.Module):
    
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(Conv2D_Norm_Activation, self).__init__()
        # pad = (kernel_size-1)//2 # kernel size is 3 or 0
        pad = padding
        self.darknetConv = nn.ModuleList()
        self.darknetConv.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, \
                                                     stride=stride, padding=pad, dilation=dilation, bias=False))
        self.darknetConv.add_module('bn', nn.BatchNorm2d(out_channels))
        self.darknetConv.add_module('leaky', nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        for dc in self.darknetConv:
            x = dc(x)
        return x

class CoordConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride = 1, padding=0, dilation=1, with_r=False, coord_conv=True):
        super().__init__()
        self.coord_conv = coord_conv
        if self.coord_conv:
            self.addcoords = AddCoords(with_r=with_r)
            # in_size = in_channels
            in_channels += 2
            if with_r:
                in_channels += 1
        self.conv = Conv2D_Norm_Activation(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


    def forward(self, x):
        if self.coord_conv:
            x = self.addcoords(x)
        # ret = x
        x = self.conv(x)
        return x