import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    #target = target.view(target.numel())
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    #print(loss)
    return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        #print(score.shape)
        return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        #print(dice_sum / class_num)
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size()[1]
        #print(logits.size(),targets.size())
        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

class scoring(nn.Module):
    def __init__(self):
        super(scoring, self).__init__()

    def forward(self, logits):
        class_num = logits.size()[1]

        score_sum = 0
        for i in range(class_num):
            S=torch.exp(logits)+1
            score=torch.sum(torch.log(S))
            score_sum += score
        return score_sum/class_num

class LwFloss(nn.Module):
    def __init__(self):
        super(LwFloss, self).__init__()

    def forward(self, oldlogits, newlogits, targets,lamta=0.5,margin=100):
        class_num = 1#logits.size()[1]
        Rt=oldlogits
        Rs=newlogits
        yreg=targets
        L_sum = 0
        for i in range(class_num):
            L1=torch.sum(abs(Rs[:,i,:,:,:]-yreg[:,i,:,:,:]))
            Lbt=torch.sum(abs(Rt[:,i,:,:,:]-yreg[:,i,:,:,:])*abs(Rt[:,i,:,:,:]-yreg[:,i,:,:,:]))
            Lbs=torch.sum(abs(Rs[:,i,:,:,:]-yreg[:,i,:,:,:])*abs(Rs[:,i,:,:,:]-yreg[:,i,:,:,:]))
            '''print(Lbs)
            print(Lbt)'''
            if (Lbs+margin>Lbt):
                L2=Lbs
            else:
                L2=0
            L=L1+lamta*L2
            L_sum+=L
        return L_sum

class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):

        #num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0.25, 0.75]).cuda()
        '''for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + 1) / (torch.sum(num_sum) + 1)'''
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)
        print(inter,union)
        return 1 - 2. * inter / union

def dice(logits, targets, class_index):
    #logits=F.softmax(logits,dim=class_index)
    #targets=F.softmax(targets,dim=1)
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    #dice = (2. * inter + 1) / (union + 1)
    dice = (2. * inter+1 ) / (union +1)
    return dice

def T(logits, targets):
    return torch.sum(targets[:, 2, :, :, :])

def P(logits, targets):
    return torch.sum(logits[:, 2, :, :, :])

def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :, :] * logits[:, 2, :, :, :])
