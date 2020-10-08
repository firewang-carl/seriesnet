# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:50:30 2020

@author: F.Wang
"""
import numpy as np
#from tqdm import tqdm
import torch
from torch.autograd import Variable
from dataload import GetPredictData

from model import  Modified3DUNet
import tifffile
#from model_unet import UNet



model1 = Modified3DUNet(in_channels=1, n_classes=3, base_n_filter = 8)
model2 = Modified3DUNet(in_channels=3, n_classes=1, base_n_filter = 8)
#model = UNet(in_dim=1, out_dim=1, num_filters=4)

#
#PATH1 = './Deepsflux_200_m1.pth'
#PATH2 = './Deepsflux_200_m2.pth'

PATH1 = './flux3d_ske_m1.pth'
PATH2 = './flux3d_ske_m2.pth'

#PATH = '/home/wx/mycode/3D/3dunet_single-chanel/Deepsflux_net400.pth'
model1.load_state_dict(torch.load(PATH1))
model2.load_state_dict(torch.load(PATH2))
model1.eval()
model2.eval()

use_gpu = torch.cuda.is_available()
if use_gpu:
     model1 = model1.cuda()   
     model2 = model2.cuda()   
     
#path=r'/home/wx/mycode/3D/dataset/0924 label(no selection)/train1/img'    #Untitled Folder   vLIDATION
#path=r'/home/wx/mycode/3D/dataset/0924 label(no selection)/train2/img'    #Untitled Folder   vLIDATION
#fruitfly
path=r'/home/wx/mycode/data/1006 flylight label/test/img'



predict_dataset = GetPredictData(path)
dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=1,shuffle=False)

DT_set=[]
theta_set=[]
phi_set=[]
count = 0
for i,data in (enumerate(dataloader, 1)): #tqdm进度条显示        
        img,name = data
        if count > 10:
            break
        count =count +1
        # cuda
        if use_gpu:
             img = img.cuda()        
        img = Variable(img)
        # 向前传播
        out = model1(img)
        pout = model2(out)
        
        out[out<0]=0
        pout[pout<0]=0
        
        
        
        DT = out[0,0]/torch.max(out[0,0]) *255
        theta = out[0,1]/torch.max(out[0,1]) *360
        phi = out[0,2]/torch.max(out[0,2]) *255
        
        ske = pout/torch.max(pout)*255
        
        DT = np.array(DT.cpu().detach().numpy())
        theta=np.array(theta.cpu().detach().numpy())
        phi=np.array(phi.cpu().detach().numpy())
        ske=np.array(ske.cpu().detach().numpy())
        
#        DT_set.append(DT)
#        theta_set.append(theta)
#        phi_set.append(phi)

        print(name)
        tifffile.imsave( r'/home/wx/mycode/data/1006 flylight label/test/test result/ske_'+name[0], (ske).astype('uint8'))
#        tifffile.imsave( r'/home/wx/mycode/3D/dataset/0924 label(no selection)/test result/DT/DT_'+name[0], (DT).astype('uint8'))
#        tifffile.imsave( r'/home/wx/mycode/3D/dataset/0924 label(no selection)/test result/theta/theta_'+name[0], (theta).astype('uint16'))
#        tifffile.imsave( r'/home/wx/mycode/3D/dataset/0924 label(no selection)/test result/phi/phi_'+name[0], (phi).astype('uint8'))

#        tifffile.imsave( r'/home/wx/mycode/3D/0825/Train/smallset/test/ske_'+name[0], (ske).astype('uint8'))
        
        
        
        