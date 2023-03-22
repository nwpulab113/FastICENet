###################################################################################################
#EDANet:Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
#Paper-Link: https://arxiv.org/ftp/arxiv/papers/1809/1809.06323.pdf
###################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math

__all__ = ["EDANet"]

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock,self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput-ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)
    
# # --- Build the EDANet Module --- #
# class EDAModule(nn.Module):
#     def __init__(self, ninput, dilated, k = 40, dropprob = 0.02):
#         super().__init__()

#         # k: growthrate
#         # dropprob:a dropout layer between the last ReLU and the concatenation of each module

#         self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
#         self.bn0 = nn.BatchNorm2d(k)

#         self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1),padding=(1,0))
#         self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3),padding=(0,1))
#         self.bn1 = nn.BatchNorm2d(k)

#         self.conv3x1_2 = nn.Conv2d(k, k, (3,1), stride=1, padding=(dilated,0), dilation = dilated)
#         self.conv1x3_2 = nn.Conv2d(k, k, (1,3), stride=1, padding=(0,dilated), dilation =  dilated)
#         self.bn2 = nn.BatchNorm2d(k)

#         self.dropout = nn.Dropout2d(dropprob)
        

#     def forward(self, x):
#         input = x

#         output = self.conv1x1(x)
#         output = self.bn0(output)
#         output = F.relu(output)

#         output = self.conv3x1_1(output)
#         output = self.conv1x3_1(output)
#         output = self.bn1(output)
#         output = F.relu(output)

#         output = self.conv3x1_2(output)
#         output = self.conv1x3_2(output)
#         output = self.bn2(output)
#         output = F.relu(output)

#         if (self.dropout.p != 0):
#             output = self.dropout(output)

#         output = torch.cat([output,input],1)
#         # print output.size() #check the output
#         return output


# --- Build the EDANet Block --- #
class EDANetBlock(nn.Module):
    def __init__(self, in_channels, num_dense_layer, dilated, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super().__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(EDAModule(_in_channels, dilated[i], growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        #self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        #out = self.conv_1x1(out)
        # out = out + x
        return out


class EDAModule(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02, ratio=2, kernel_size=1, dw_size=3, stride=1, relu=True):
        super().__init__()

        self.out = k
        init_channels = math.ceil(self.out / ratio)

        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        # print("kernel: ", kernel_size)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(ninput, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())

        #  convs.append(nn.Conv2d(inter_channel, inter_channel, kernel_size=3, stride = 1, padding=dilated, dilation=dilated, bias=False))
        # 第二次卷积：注意有个参数groups，为分组卷积
        # 每个feature map被卷积成 raito - 1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dilated, dilation=dilated, groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        input = x
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        output = torch.cat([x1, x2], dim=1)
        output = torch.cat([output, input], 1)
        # print output.size() #check the output
        return output


class EDANet(nn.Module):
    def __init__(self, classes=19):
        super(EDANet,self).__init__()

        self.layers1 = nn.ModuleList()

        # DownsamplerBlock1
        self.layers1.append(DownsamplerBlock(3, 15))


        self.layers2 = nn.ModuleList()
        # DownsamplerBlock2
        self.layers2.append(DownsamplerBlock(15, 30)) 
        self.layers2.append(DownsamplerBlock(30, 60))
        # EDA Block1
        self.layers2.append(EDANetBlock(60, 5, [1,1,1,2,2], 20))

        self.layers3 = nn.ModuleList()
        # DownsamplerBlock3
        self.layers3.append(DownsamplerBlock(160, 160))
        # # EDA Block2
        self.layers3.append(EDANetBlock(160, 8, [2,2,4,4,8,8,16,16], 20))
        # 160 160 320
        # Projection layer
        # self.project_layer = nn.Conv2d(450,classes,kernel_size = 1)
        
        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):

        output1 = x

        
        for layer in self.layers1:
            output1 = layer(output1)
        

        output2 = output1 
        for layer in self.layers2:
            output2 = layer(output2)

        
        output3 = output2
        for layer in self.layers3:
            output3 = layer(output3)
        # print("hell V8")
        # print(output1.shape, output2.shape, output3.shape)
        # 1/2 1/4 1/8
        # 15 160 320
        return output1, output2, output3





"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDANet(classes=19).to(device)
    summary(model,(3,800,800))
