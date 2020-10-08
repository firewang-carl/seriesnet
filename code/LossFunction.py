 # -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:47:51 2020
自定义加权损失函数（LS）,权重为像素类别
@author: F.Wang
"""
# from torchvision import transforms
import torch
# from torch.autograd import Variable
from dataload import GetData
import numpy as np
import torch.nn as nn

def weightmap(image):
    weight=np.array(image)
    # weight=image
    Rsc=np.count_nonzero(image)  #前景个数
    Rb=weight.size-Rsc     #背景个数
    wcs=Rb/weight.size
    wb=Rsc/weight.size
    weight[weight==1]=wcs      #weight>0是前景
    weight[weight==0]=wb      #weight=0是背景
#    print("wcs:{} wb:{}".format(wcs,wb))
    return torch.from_numpy(weight)
    


#def weightmap(image):
#
#    weight = torch.zeros_like(image)
##    Rsc = image.nonzero().shape[0]
#    Rb  =  image[image==0].shape[0]     #背景个数
#    wcs=Rb/image.numel()    # 前景权重
##    wb=Rsc/image.numel()  # 背景
#    wb =1-wcs
#    weight[image==1]=wcs      #weight>0是前景
#    weight[image==0]=wb      #weight=0是背景
#    
##    print("wcs:{} wb:{}".format(wcs,wb))
#
#    return weight
    


"根据权重计算loss"


    
#def lossfunc(out,label,weight):
#      criterion = nn.MSELoss()
##      criterion = nn.MSELoss(reduction='none')
##      weight =weightmap(weight)
##      error=criterion(label,out)*weight
#     
#      error=criterion(label,out)
##     
#      return error.mean()
##     return error.sum()
##    return error


def lossfunc(out,label,weight):

    criterion = nn.MSELoss(reduction='none')
#    weight =weightmap(weight)
    error=criterion(label,out)*weight

    return error.mean()




if __name__=="__main__":  #测试损失函数E    
    full_dataset = GetData(path0=r'/home/wx/mycode/3D/dataset/0912 label/train/img',
                            path1=r'/home/wx/mycode/3D/dataset/0912 label/train/DT label', 
                            path2=r'/home/wx/mycode/3D/dataset/0912 label/train/class label')
    
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, vald_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,drop_last =True, shuffle=True)
    vald_loader = torch.utils.data.DataLoader(vald_dataset, batch_size=1,drop_last =True, shuffle=True)
    
    
    
    
    for i, data in (enumerate(train_loader, 1)): #tqdm进度条显示        
#            pic, label,classlabel,img0= data
            pic,label,classlabel =data
#            weight=weightmap(classlabel)
            loss = lossfunc(label,pic,classlabel)
            print(loss)
            break






#      a= torch.tensor([[3.,3.],[4.,4.]])
#      b= torch.tensor([[1.,1.],[2.,2.]])
#      criterion = nn.MSELoss(reduction='none')
# #     criterion = nn.MSELoss(reduction='sum')
#      error=criterion(a,b)
#      print(error)