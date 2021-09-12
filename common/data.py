import random
import torch
import   cv2 as cv
from   imgaug           import augmenters as iaa
from   torchvision      import transforms

class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)
        
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

    # Load Segmentation data
    def loadSeg(self, idx):
        # Read GT
        seg = cv.imread(self.filepaths[idx]['seg'], cv.IMREAD_COLOR)
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
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        return {'img': self.loadImage(idx),
                'seg': self.loadSeg  (idx)}