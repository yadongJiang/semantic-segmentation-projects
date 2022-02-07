import torch
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self, in_channs, out_channs, kernel_size, stride, padding=None):
        super(CBR, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channs, out_channs, kernel_size=kernel_size, \
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channs)
        self.act = nn.PReLU(out_channs)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InputProjectionA(nn.Module):
    def __init__(self, smaplingTimes):
        super(InputProjectionA, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(smaplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    
    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x

class BR(nn.Module):
    def __init__(self, out_channs):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_channs, eps=1e-03)
        self.act = nn.PReLU(out_channs)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x

class C(nn.Module):
    def __init__(self, in_channs, out_channs, kernel_size, stride, padding=None):
        super(C, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channs, out_channs, kernel_size=kernel_size, 
                                stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)

class CDilated(nn.Module):
    def __init__(self, in_channs, out_channes, kernel_size, stride, dilate):
        super(CDilated, self).__init__()
        padding = (kernel_size // 2) * dilate
        self.conv = nn.Conv2d(in_channs, out_channes, kernel_size=kernel_size, \
                                stride=stride, padding=padding, dilation=dilate, bias=False)

    def forward(self, x):
        return self.conv(x)

class DownSamplerB(nn.Module):
    def __init__(self, in_channs, out_channs):
        super(DownSamplerB, self).__init__()
        n = int(out_channs / 5)
        n1 = out_channs - n*4
        self.c1 = C(in_channs, n, 3, 2) # 在平行dialte convs之前，首先降维
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(out_channs, eps=1e-3)
        self.act = nn.PReLU(out_channs)
    
    def forward(self, x):
        output1 = self.c1(x)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
        output = self.bn(combine)
        output = self.act(output)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, in_channs, out_channs, add=True):
        super(DilatedParllelResidualBlockB, self).__init__()
        n = int(out_channs/5) if out_channs >= 20 else out_channs
        n1 = out_channs - 4*n if out_channs >= 20 else out_channs
        # print("in_channs: ", in_channs, "out_channs: ", out_channs, "n: ", n, " n1: ", n1)
        self.c1 = C(in_channs, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)

        if out_channs<20:
            self.conv = nn.Conv2d(5 * out_channs, out_channs, 1, 1, bias=False)
        else:
            self.conv = None
        self.bn = BR(out_channs)
        self.add = add

    def forward(self, x):
        output1 = self.c1(x)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
        if self.conv:
            combine = self.conv(combine)
        if self.add:
            combine = combine + x
        output = self.bn(combine)
        return output