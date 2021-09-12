
import os
import glob
import random
import numpy as np

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
    random.shuffle( filepaths )
    return filepaths

def getPart(data,isTrain):
    n = len(data)
    a = int(0.8*n)
    if isTrain: return data[:a]
    else      : return data[a:]