# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:57:36 2020
本模块自定义数据加载及数据增强
 3D model needs 5 pics with 3-dimension size
    image
    classlabel
    DTlabel
    thetalabel
    philabel
@author: F.Wang
"""
import torch
from torch.utils.data import Dataset
import os  #文件读取
import tifffile
import numpy as np
from tqdm import tqdm




def getFiles(dirp, suffix='.tif'): # 查找根目录，文件后缀 
   res = []
   for root, directory, files in os.walk(dirp):  # =>当前根,根下目录,目录下的文件
     for filename in files:
        name, suf = os.path.splitext(filename) # =>文件名,文件后缀
        if suf == suffix:
            # res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
             res.append(filename)
   res.sort()
   return res

class GetData(Dataset):
    def __init__(self,path0,path1,path2,path3,path4,path5): #得到名字list   path0为路径名

        super(GetData,self).__init__()
        self.path0 = path0
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4
        self.path5 = path5
        
        self.name0_list = getFiles(self.path0)  #os.listdir把目标路径里的全部文件以list方式列出里
        self.name1_list = getFiles(self.path1)
        self.name2_list = getFiles(self.path2)
        self.name3_list = getFiles(self.path3)
        self.name4_list = getFiles(self.path4)
        self.name5_list = getFiles(self.path5)        
        
        
    def __len__(self):
        return len(self.name0_list)

    def __getitem__(self, index): #按名取图,index对应批次
          
        self.name0 = self.name0_list[index]
        self.name1 = self.name1_list[index]
        self.name2 = self.name2_list[index]
        self.name3 = self.name3_list[index]
        self.name4 = self.name4_list[index]
        self.name5 = self.name5_list[index]
        #读取tif用tifffile

        img0 = tifffile.imread(os.path.join(self.path0, self.name0))     #pic
        img1 = tifffile.imread(os.path.join(self.path1, self.name1))     #DT
        img2 = tifffile.imread(os.path.join(self.path2, self.name2))     #theta
        img3 = tifffile.imread(os.path.join(self.path3, self.name3))     #phi
        img4 = tifffile.imread(os.path.join(self.path4, self.name4))     #ske
        img5 = tifffile.imread(os.path.join(self.path5, self.name5))     #class
        
        #        img3 = np.flip(img3,axis=1)
        
        #flylight /mouse
        pic = np.zeros((1,32,64,64),dtype='float32')
        DTlabel  = np.zeros((1,32,64,64),dtype='float32')
        thetalabel  = np.zeros((1,32,64,64),dtype='float32')
        philabel = np.zeros((1,32,64,64),dtype='float32')
        skelabel  = np.zeros((1,32,64,64),dtype='float32')
        classlabel = np.zeros((1,32,64,64),dtype='float32')
        # fruitfly
#        pic = np.zeros((1,16,64,64),dtype='float32')
#        DTlabel  = np.zeros((1,16,64,64),dtype='float32')
#        thetalabel  = np.zeros((1,16,64,64),dtype='float32')
#        philabel = np.zeros((1,16,64,64),dtype='float32')
#        skelabel  = np.zeros((1,16,64,64),dtype='float32')
#        classlabel = np.zeros((1,16,64,64),dtype='float32')
        
#        pic[0] = img0/255
        
        pic[0] = img0/np.max(img0)
        DTlabel[0] =img1/255        
        thetalabel[0] = img2/360
        philabel[0]  = img3/255
        skelabel[0] =img4/255
        classlabel[0] =img5/255
       

        label =np.concatenate((DTlabel,thetalabel,philabel), axis=0)
#        print(label.shape)
        # return pic,DTlabel,thetalabel,philabel,classlabel,label
        return pic,label,skelabel,classlabel


class GetPredictData(Dataset):
    def __init__(self,path0): #得到名字list   path0为路径名

        super(GetPredictData,self).__init__()
        self.path0 = path0
        self.name0_list = getFiles(self.path0)  #os.listdir把目标路径里的全部文件以list方式列出里
#        self.img2data = transforms.Compose([           #数据增强
#                            # transforms.RandomResizedCrop(224),
##                            transforms.Resize([256,256]),
#                            transforms.ToTensor(),
#                            # transforms.Normalize(mean = [ 0.485],    
#                            #                       std  = [ 0.229]),
#                            ])
       
        
    def __len__(self):
        return len(self.name0_list)

    def __getitem__(self, index): #按名取图,index对应批次
          
        self.name0 = self.name0_list[index]

        #读取tif用tifffile
        img0 = tifffile.imread(os.path.join(self.path0, self.name0))
        
        
        pic = np.zeros((1,32,64,64),dtype='float32')
        
        
#        pic = np.zeros((1,16,64,64),dtype='float32')
#        pic[0] = img0/np.max(img0)
        pic[0] = img0/np.max(img0)
        
#        imgdata0 = self.img2data(img0)
        return pic,self.name0

if __name__ == "__main__":
    
    
#    full_dataset = GetData(path0=r'/home/wx/mycode/3D/dataset/0921 label/train/img',
#                            path1=r'/home/wx/mycode/3D/dataset/0921 label/train/DT label', 
#                            path2=r'/home/wx/mycode/3D/dataset/0921 label/train/theta label',
#                            path3=r'/home/wx/mycode/3D/dataset/0921 label/train/phi label',
#                            path4=r'/home/wx/mycode/3D/dataset/0921 label/train/prob label',
#                            path5=r'/home/wx/mycode/3D/dataset/0921 label/train/class label')

#    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1,drop_last =True, shuffle=True)

#  fruitfly
#    full_dataset = GetData(path0=r'/home/wx/mycode/data/0927 fruitfly data/img',
#                            path1=r'/home/wx/mycode/data/0927 fruitfly data/DT label', 
#                            path2=r'/home/wx/mycode/data/0927 fruitfly data/theta label',
#                            path3=r'/home/wx/mycode/data/0927 fruitfly data/phi label',
#                            path4=r'/home/wx/mycode/data/0927 fruitfly data/swc_mask',
#                            path5=r'/home/wx/mycode/data/0927 fruitfly data/class label')

    full_dataset = GetData(path0=r'/home/wx/mycode/data/1006 flylight label/img',
                            path1=r'/home/wx/mycode/data/1006 flylight label/DT label', 
                            path2=r'/home/wx/mycode/data/1006 flylight label/theta label',
                            path3=r'/home/wx/mycode/data/1006 flylight label/phi label',
                            path4=r'/home/wx/mycode/data/1006 flylight label/ske label',
                            path5=r'/home/wx/mycode/data/1006 flylight label/class label')
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1,drop_last =True, shuffle=True)    
    
    
    for i, data in tqdm(enumerate(train_loader, 1)): #tqdm进度条显示        
#            pic, label,classlabel,img0= data
            pic,label,skelabel,classlabel =data
#            print(pic,label,classlabel )
#            break
            print(pic.shape)
#             print("0")
            break
#            name0,name1,name2 = data
#            print(pic[0,0])
#            print(classlabel[0,0])
#            print(i,name0,name1,name2)

