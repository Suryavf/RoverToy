import os
import glob
import random
import  numpy as np
import pandas as pd
import    cv2 as cv
from      tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision      import transforms
from   imgaug           import augmenters as iaa
from   torch.utils.data import DataLoader
from   torchsummary import summary

def getint(name):
    basename = name.partition('.')
    _, num = basename[0].split('resume')
    return int(num)

def checkdirectory(d):
    if not os.path.exists(d):
        os.makedirs(d)

class averager():
    def __init__(self,n=0):
        if n >0:
            self.mean = np.array(n)
        else:    
            self.mean  = 0
        self.count = 0
        self.n = n
    def reset(self):
        if self.n >0:
            self.mean = np.array(self.n)
        else:    
            self.mean  = 0
        self.count = 0
    def update(self,val):
        n = self.count
        self.count = n + 1
        self.mean  = (self.mean*n + val)/self.count
    def val(self):
        return self.mean



# =========================================================================================================

class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)
        

def getFilesPath(datapath):
    folderpaths = glob.glob(os.path.join(datapath, 'CKA_16*'))
    folderpaths.sort()
    filepaths = []
    for f in folderpaths:
        ss = glob.glob(os.path.join(f,'annotations/dlp/colorCleaned','*.png'))
        ss.sort()

        for s in ss:
            name = os.path.basename(s)
            i = os.path.join(f,'images/rgb',name)
            # Check
            if not os.path.exists(i): continue
            filepaths.append({'img':i,'seg':s})

    p = np.random.permutation(len(filepaths))
    return filepaths[p]

def getPart(data,isTrain):
    n = len(data)
    lim = int(0.8*n)

    if isTrain: data[:lim]
    else      : data[lim:]


class SugarDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, isTrain):
        # Parameters
        self.filepaths = filepaths
        self.isTrain = isTrain
        self.H = 384
        self.W = 512
        
        # Data augmentation -> Img
        self.transformDataAug = transforms.RandomOrder([
                  RandomTransWrapper( seq=iaa.GaussianBlur((0, 1.5)),
                                      p=0.09),
                  RandomTransWrapper( seq=iaa.AdditiveGaussianNoise(loc=0,scale=(0.0, 0.05),per_channel=0.5),
                                      p=0.09),
                  RandomTransWrapper( seq=iaa.Add((-20, 20), per_channel=0.5),
                                      p=0.3),
                  RandomTransWrapper( seq=iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                      p=0.4),
                  RandomTransWrapper( seq=iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),
                                      p=0.09),
                  ])

        trans = list()
        if   self.isTrain: trans.append(self.transformDataAug)
        trans.append(transforms.ToPILImage())
        trans.append(transforms.Resize( [self.H,self.W] ))
        trans.append(transforms.ToTensor())
        self.imgTransform = transforms.Compose(trans)

        # Segmentation
        trans = list()
        trans.append(transforms.ToPILImage())
        trans.append(transforms.Resize( [self.H,self.W] ))
        trans.append(transforms.ToTensor())
        self.segTransform = transforms.Compose(trans)

    def getFilesPath(self, datapath, isTrain):
        folderpaths = glob.glob(os.path.join(datapath, 'CKA_16*'))
        folderpaths.sort()
        folderpaths = folderpaths[:17] if isTrain else folderpaths[17:19]
        imgpath = []
        segpath = []

        for f in folderpaths:
            ss = glob.glob(os.path.join(f,'annotations/dlp/colorCleaned','*.png'))
            ss.sort()

            for s in ss:
                name = os.path.basename(s)
                i = os.path.join(f,'images/rgb',name)

                # Check
                if not os.path.exists(i): continue
                imgpath.append(i)
                segpath.append(s)
        return imgpath,segpath


    # Load Segmentation data
    def loadSeg(self, idx):
        # Read GT
        seg = cv.imread(self.segPath[idx]['seg'], cv.IMREAD_COLOR)
        seg = self.segTransform(seg)

        rock  = (seg[2]>0)
        plant = (seg[1]>0)
        backg = ~torch.logical_or(rock,plant)
        
        seg[0] = backg
        seg[1] = plant
        seg[2] = rock
        return seg

    # Load image
    def loadImage(self, idx):
        img = cv.imread(self.filepaths[idx]['img'], cv.IMREAD_COLOR)
        img = self.imgTransform(img)
        return img
    

    def __len__(self):
        return len(self.imgPath)
    
    def __getitem__(self, idx):
        return {'img': self.loadImage(idx),
                'seg': self.loadSeg  (idx)}



# =========================================================================================================

from torchvision.models import mobilenet_v3_small as mobilenet3s
from collections import OrderedDict
from torch import nn
from typing import Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class DeepLabV3(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



class MobilNet3DeepLab3(nn.Module):
    def __init__(self,n_classes):
        super(MobilNet3DeepLab3,self).__init__()

        mod = mobilenet3s(pretrained=True)
        backbone = mod.features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
        # ---------------------------------------------
        return_layers = {out_layer: 'out'}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        classifier = DeepLabHead(out_inplanes, n_classes)
        self.model = DeepLabV3(backbone,classifier,None)

    def forward(self,x):
        y = self.model(x)
        return y['out']






# =========================================================================================================



import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class Main(object):
    """ Constructor """
    def __init__(self,model,imgpath="SugarBeet/ijrr_sugarbeets_2016_annotations",
                            outpath="",
                            n_epoch=500):
        # Parameters
        self.outpath = outpath
        self.n_epoch = n_epoch
        self.isVAE    = False
        self.multiBCE = True
        self.BCE      = False
    
        self.crvTrain = list()
        self.crvTest  = list()
        
        self.model  = model
        self.device = torch.device('cuda:0')
        self.model  = self.model.to(self.device)

        self.lossfun = DiceLoss()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00001, 
                                       weight_decay=1e-8, momentum=0.9)
        
        # Dataset
        paths = getFilesPath(imgpath)
        self.trainDataset = DataLoader(SugarDataset(getPart(paths,isTrain=True),
                                                    isTrain=True),batch_size=4)
        self.evalDataset  = DataLoader(SugarDataset(getPart(paths,isTrain=False),
                                                    isTrain=False),batch_size=4)
        
    """ Training state functions """
    def state_reset(self):
        self._state = {}
    def state_add(self,name,attr):
        self._state[name]=attr

    def state_save(self,epoch):
        # Save model
        pathMod = os.path.join( self.outpath, "model" + str(epoch) + ".pth" )
        torch.save( self._state, pathMod)


    """ Training """
    def train(self):
        # Parameters
        running_loss = 0.0
        stepView     =  10
        dataloader   = self.trainDataset
        lossTrain    = averager()
        
        self.model.train()
        with tqdm(total=len(dataloader),leave=False) as pbar:
            for i, sample in enumerate(dataloader):
                # Batch
                dev_img = sample['img'].to(self.device)
                dev_seg = sample['seg'].to(self.device)
                
                # Model
                dev_pred = self.model(dev_img)
                dev_loss = self.lossfun(dev_pred, dev_seg)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.model    .zero_grad()
                
                dev_loss.backward()
                self.optimizer.step()

                # Print statistics
                runtime_loss = dev_loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchTrain loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossTrain.update(runtime_loss)
                pbar. update()
                pbar.refresh()
            pbar.close()

        lossTrain = lossTrain.val()
        print("Epoch training loss:",lossTrain)
        return lossTrain

    """ Evaluation """
    def eval_(self):
        # Parameters
        running_loss = 0.0
        stepView     =  10
        dataloader   = self.evalDataset
        lossEval     = averager()

        # Model to evaluation
        self.model.eval()
        with torch.no_grad(), tqdm(total=len(dataloader),leave=False) as pbar:
            for i, sample in enumerate(dataloader):
                # Batch
                dev_img = sample['img'].to(self.device)
                dev_seg = sample['seg'].to(self.device)
                
                # Model
                dev_pred = self.model(dev_img)
                dev_loss = self.lossfun(dev_pred, dev_seg)

                # Print statistics
                runtime_loss = dev_loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchEval loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossEval.update(runtime_loss)
                pbar. update()
                pbar.refresh()
            pbar.close()

        lossEval = lossEval.val()
        print("Epoch evaluation loss:",lossEval)
        return lossEval

    """ Training """
    def execute(self):
        # Parameters
        n_epoch = self.n_epoch
        
        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("\nEpoch",epoch+1,"-"*40)

            # Train
            lossTrain = self.train()
            lossEval  = self.eval_()

            # Save
            self.crvTrain.append(lossTrain)
            self.crvTest .append(lossEval )

            # Save checkpoint
            self.state_add (    'model',self.    model.state_dict())
            self.state_add ('optimizer',self.optimizer.state_dict())
            self.state_save(epoch+1)


    def plot(self):
        dataloader = self.evalDataset

        # Model to evaluation
        self.model.eval()
        with torch.no_grad(), tqdm(total=len(dataloader),leave=False) as pbar:
            for i, sample in enumerate(dataloader):
                # Batch
                hst_img = sample['img']
                dev_img = hst_img.to(self.device)
                dev_seg = sample['seg']
                
                # Model
                d0, d1, d2, d3, d4, d5, d6 = self.model(dev_img)
                dev_pred = torch.sigmoid(d0)

                output = (dev_pred > 0.5).data.cpu().numpy()
                output = output.squeeze()*255
                
                # Real
                real = dev_seg.data.cpu().numpy().squeeze()*255
                img  = hst_img.data.cpu().numpy().squeeze().transpose([1,2,0])*255
                
                out = np.zeros([240,320, 3])
                out[:,:,0] = real
                out[:,:,2] = output

                cvimshow(np.hstack([img,out]))



# =========================================================================================================




path = "/home/victor/sugar/ijrr_sugarbeets_2016_annotations"
model = MobilNet3DeepLab3(3)
obj = Main(model,imgpath=path,n_epoch=100)

obj.execute()

