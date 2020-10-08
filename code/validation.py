#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:27:35 2020

@author: wx
"""


# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:59:31 2020

@author: F.Wang
"""
import torch
import torch.nn as nn

#from tqdm import tqdm
from torch.autograd import Variable
#from VGG16_ASPP import VGG_ASPP,_initialize_weights
import LossFunction
import numpy as np
# import matplotlib.pyplot as plt
#import os
#from model import unet.UNet
#from unet import  UNet



#'''定义超参数'''
#
#batch_size = 1       # 批的大小
#learning_rate =1e-1   # 学习率1e-4
##learning_rate =1   # 学习率1e-4
#num_epoches =100 # 遍历训练集的次数
#
#
#
#
#''' 数据加载 '''
#
#
#
#
#vali_dataset = GetData(path0=r'/home/wx/wxcode/traindata/512/I',
#                         path1=r'/home/wx/wxcode/traindata/512//M', 
#                         path2=r'/home/wx/wxcode/traindata/512/C')
#
#
#
#
#vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=1,drop_last =True, shuffle=True)
#'''创建model实例对象，并检测是否支持使用GPU'''
#
#model = UNet()
#
##    load pretrained model
#PATH = './Deepsflux_net（512DT）.pth'
#model.load_state_dict(torch.load(PATH))
#
#
#model.eval()
#
#use_gpu = torch.cuda.is_available()
#if use_gpu:
#     model = model.cuda()     
#print('use_gpu',use_gpu)



def vali_mymodel(model1,model2,vali_loader):
    use_gpu = torch.cuda.is_available() 
    '''验证模型'''
#    count = 0
    running_loss=0
#    with torch.no_grad():
    model1.eval()
    model2.eval()
    for i, data in (enumerate(vali_loader, 1)): 
        img, label,skelabel,classlabel= data

        if use_gpu:
           img =img.cuda()   
           label = label.cuda()
           skelabel = skelabel.cuda()
#           classlabel = classlabel.cuda()
        img = Variable(img)
        label = Variable(label)
        skelabel =   Variable(skelabel)
    
         # 向前传播
#        with torch.no_grad():
        out = model1(img)
        pout = model2(out)
        weight=LossFunction.weightmap(classlabel[0,0])
        weight=weight.cuda()
        
        loss1=LossFunction.lossfunc(out,label,weight)
        loss2=LossFunction.lossfunc(pout,skelabel,weight)      
        
        loss =loss1+loss2
        if np.isnan(loss.item()):
              print('Loss value is NaN!')

        running_loss += loss.item()     
#        count =count+1
    vali_loss = running_loss / len(vali_loader)    
    return vali_loss
    
#print('%.10f'%vali_mymodel(model,vali_loader))
#
#    print('Finish {} epoch, Loss: {:.10f}'.format(
#        epoch + 1, running_loss / (len(train_dataset))))
    
#print('Validaion error:{:.10f}'.format(vali_mymodel(model,vali_loader)))