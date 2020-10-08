# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:59:31 2020

@author: F.Wang
"""
import torch
from torch import  optim
from torch.autograd import Variable

import LossFunction
import numpy as np
import validation
import os
from model import  Modified3DUNet

from dataload import GetData

'''定义超参数'''

batch_size = 1       # 批的大小
learning_rate =1e-1   # 学习率1e-4
num_epoches = 300 # 遍历训练集的次数

''' 数据加载 '''
#set train data path
#full_dataset = GetData(path0=r'/home/wx/mycode/data/0927 fruitfly data/img',
#                            path1=r'/home/wx/mycode/data/0927 fruitfly data/DT label', 
#                            path2=r'/home/wx/mycode/data/0927 fruitfly data/theta label',
#                            path3=r'/home/wx/mycode/data/0927 fruitfly data/phi label',
#                            path4=r'/home/wx/mycode/data/0927 fruitfly data/swc_mask',
#                            path5=r'/home/wx/mycode/data/0927 fruitfly data/class label')

#flylight

full_dataset = GetData(path0=r'/home/wx/mycode/data/1006 flylight label/train/img',
                            path1=r'/home/wx/mycode/data/1006 flylight label/train/DT label', 
                            path2=r'/home/wx/mycode/data/1006 flylight label/train/theta label',
                            path3=r'/home/wx/mycode/data/1006 flylight label/train/phi label',
                            path4=r'/home/wx/mycode/data/1006 flylight label/train/ske label',
                            path5=r'/home/wx/mycode/data/1006 flylight label/train/class label')

train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, vald_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,drop_last =True, shuffle=True)
vald_loader = torch.utils.data.DataLoader(vald_dataset, batch_size=1,drop_last =True, shuffle=True)

'''创建model实例对象，并检测是否支持使用GPU'''

model1 = Modified3DUNet(in_channels=1, n_classes=3, base_n_filter = 8)
model2 = Modified3DUNet(in_channels=4, n_classes=1, base_n_filter = 8)

#    load pretrained model
#PATH1 = './flux3d_ske_m1.pth'
#PATH2 = './flux3d_ske_m2.pth'
#model1.load_state_dict(torch.load(PATH1))
#model2.load_state_dict(torch.load(PATH2))


use_gpu = torch.cuda.is_available()
if use_gpu:
     model1 = model1.cuda()     
     model2 = model2.cuda()    
print('use_gpu',use_gpu)

'''定义loss和optimizer'''

optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.2, last_epoch=-1)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.2, last_epoch=-1)
#milestones =[25,50]
#scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
'''训练模型'''


for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)      # .format为输出格式，formet括号里的即为左边花括号的输出
    running_loss = 0.0
    count=0
    model1.train()
    model2.train()
    for i, data in (enumerate(train_loader, 1)): 
        img, label,skelabel, classlabel= data

        if use_gpu:
           img =img.cuda()   
           label = label.cuda()
           skelabel = skelabel.cuda()
#           classlabel = classlabel.cuda()
        img = Variable(img)
        label = Variable(label)
        skelabel = Variable(skelabel)
#        classlabel = Variable(classlabel)

         # 向前传播
#        with torch.no_grad():
        out = model1(img)
        
        pint =torch.cat((img,out),1)
        pout = model2(pint)  # add new chanel with img
#        pout = model2(out)

        optimizer1.zero_grad()  #清除每个批次的梯度·
        optimizer2.zero_grad()  #清除每个批次的梯度·
        
        
        weight=LossFunction.weightmap(classlabel[0,0])
        weight=weight.cuda()
        
        
        loss1=LossFunction.lossfunc(out,label,weight)
        loss2=LossFunction.lossfunc(pout,skelabel,weight)
        loss =loss1+loss2
        
#        print(loss.item())
        if np.isnan(loss.item()):
              print('Loss value is NaN!')

        running_loss += loss.item()
        # # 向后传播       
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
    # update learning rate outside the inner iteration
    scheduler1.step()
    scheduler2.step()    
        
    
    print('Finish {} epoch, Loss: {:.10f}'.format(
        epoch + 1, running_loss / len(train_dataset )))
    
    filename = './write_loss.txt'
    with open(filename,'a') as f: # 如果filename不存在会自动创建， 
             f.write('Finish {} epoch, Loss: {:.10f}\n'.format(
        epoch + 1, running_loss / len(train_dataset)))
             
##      save the parameter
##    if epoch+1 >= 100 :
    if (epoch+1)%50 == 0:    
            PATH1 = './Deepsflux_{}_m1.pth'.format(epoch + 1)
            PATH2 = './Deepsflux_{}_m2.pth'.format(epoch + 1)
            torch.save(model1.state_dict(), PATH1)
            torch.save(model2.state_dict(), PATH2)
#    
    """  validaiton  """
    if (epoch+1)%5 == 0:  
        vali_loss = validation.vali_mymodel(model1,model2,vald_loader)
        print('Validaion error:{:.10f}'.format(vali_loss))
        with open(filename,'a') as f: # 如果filename不存在会自动创建， 
           f.write('Validaion error:{:.10f}\n'.format(vali_loss))


PATH1 = './flux3d_ske_m1.pth'
PATH2 = './flux3d_ske_m2.pth'
torch.save(model1.state_dict(), PATH1)
torch.save(model2.state_dict(), PATH2)

os.system('shutdown')