from torch import nn
from torch import cat


class pub(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True,dropout=False):
        super(pub, self).__init__()
        inter_channels = out_channels if in_channels > out_channels else out_channels//2
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    #nn.Dropout3d(p=0.5),
                    nn.LeakyReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    #nn.Dropout3d(p=0.5),
                    nn.LeakyReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.InstanceNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.InstanceNorm3d(out_channels))
        if dropout:
            layers.insert(len(layers) - 1, nn.Dropout3d(p=0.5))
        self.pub = nn.Sequential(*layers)
    def forward(self, x):
        return self.pub(x)


class unet3dEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True,dropout=False):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm,dropout)
        self.pool = nn.MaxPool3d(2, stride=2)
    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='trilinear')
            #self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
    def forward(self, x, x1):
        x = self.sample(x)
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, in_channels, filter_num_list,class_nums):
        super(unet3d, self).__init__()
        self.in_channels = in_channels
        self.class_nums = class_nums
        batch_norm = True
        sample = False
        dropout=False
        self.en1 = unet3dEncoder(self.in_channels, filter_num_list[0], batch_norm)
        self.en2 = unet3dEncoder(filter_num_list[0], filter_num_list[1], batch_norm)
        self.en3 = unet3dEncoder(filter_num_list[1], filter_num_list[2], batch_norm)
        self.en4 = unet3dEncoder(filter_num_list[2], filter_num_list[3], batch_norm)
        self.en5 = unet3dEncoder(filter_num_list[3], filter_num_list[4], batch_norm,dropout=True)
        self.up4 = unet3dUp(filter_num_list[4], filter_num_list[3], batch_norm, sample)
        self.up3 = unet3dUp(filter_num_list[3], filter_num_list[2], batch_norm, sample)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1], batch_norm, sample)
        self.up1 = unet3dUp(filter_num_list[1], filter_num_list[0], batch_norm, sample)
        self.con_last = nn.Conv3d(filter_num_list[0], class_nums, kernel_size=1)
        self.softmax = nn.Sigmoid()


    def forward(self, x):
        x1,x = self.en1(x)
        print(x1.shape,x.shape)
        x2,x = self.en2(x)
        print(x2.shape,x.shape)
        x3,x = self.en3(x)
        print(x3.shape,x.shape)
        x4,x = self.en4(x)
        print(x4.shape,x.shape)
        x5,_ = self.en5(x)
        print(x5.shape,x.shape)
        x = self.up4(x5, x4)
        print(x.shape)
        x = self.up3(x, x3)
        print(x.shape)
        x = self.up2(x, x2)
        print(x.shape)
        x = self.up1(x, x1)
        print(x.shape)
        out = self.con_last(x)
        print(out.shape)
        #out[:,0]=self.softmax(out[:,0]) #2021.10.13 bad_resize
        #out[:,1] = self.softmax(out[:,1]) #2021.10.13 bad_resize
        #out[0]=self.softmax(out[0])
        #out[1] = self.softmax(out[1])
        out=self.softmax(out)
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()