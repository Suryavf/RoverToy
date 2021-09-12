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

import common.utils as U
import common.loss as L
from   common.data import SugarDataset
from   network.MobilNeASPP import MobilNet3DeepLab3

import torch
import torch.optim as optim
from   torch.utils.data import DataLoader


class Main(object):
    """ Constructor """
    def __init__(self,model,imgpath="SugarBeet/ijrr_sugarbeets_2016_annotations",
                            wightpath='',
                            outpath="",
                            n_epoch=500):
        # Parameters
        self.outpath = outpath
        self.n_epoch = n_epoch
        self.isVAE    = False
        self.multiBCE = True
        self.BCE      = False
        self.state    = {}
    
        self.crvTrain = list()
        self.crvTest  = list()
        
        self.model  = model
        self.device = torch.device('cuda:0')
        self.model  = self.model.to(self.device)

        self.lossfun = L.DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        
        # Dataset
        paths = U.getFilesPath(imgpath)
        self.trainDataset = DataLoader(SugarDataset(U.getPart(paths,isTrain=True),
                                                    isTrain=True),batch_size=50)
        self.evalDataset  = DataLoader(SugarDataset(U.getPart(paths,isTrain=False),
                                                    isTrain=False),batch_size=50)
        
    """ Training state functions """
    def state_reset(self):
        self.state = {}
    def state_add(self,name,attr):
        self.state[name]=attr

    def state_save(self,epoch):
        # Save model
        pathMod = os.path.join( self.outpath, "model" + str(epoch) + ".pth" )
        torch.save( self.state, pathMod)
    
    """ Load model """
    def load(self,path):
        # Load
        checkpoint = torch.load(path)
        self.model    .load_state_dict(checkpoint[    'model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    """ Training """
    def train(self):
        # Parameters
        running_loss = 0.0
        stepView     =  10
        dataloader   = self.trainDataset
        lossTrain    = U.averager()
        
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
        lossEval     = U.averager()

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
                dev_img = sample['img'].to(self.device)
                dev_seg = sample['seg'].to(self.device)
                
                # Model
                dev_pred = self.model(dev_img)

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

