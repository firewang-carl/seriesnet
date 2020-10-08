#import torch
import numpy as np
import tifffile
import sys
from tqdm import tqdm
#import predict





def predict_block(block) :
    import numpy as np
    import torch
    from torch.autograd import Variable
#    from dataload import GetPredictData
    from model import  Modified3DUNet
    
    
    # load model
    model1 = Modified3DUNet(in_channels=1, n_classes=3, base_n_filter = 8)
    model2 = Modified3DUNet(in_channels=3, n_classes=1, base_n_filter = 8)

    PATH1 = './flux3d_ske_m1.pth'
    PATH2 = './flux3d_ske_m2.pth'
    
    model1.load_state_dict(torch.load(PATH1))
    model2.load_state_dict(torch.load(PATH2))
    model1.eval()
    model2.eval()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
         model1 = model1.cuda()   
         model2 = model2.cuda()   
         

    # convert the block to standarded torch
    nz,ny,nx = block.shape
    pic = np.zeros((1,1,nz,ny,nx),dtype='float32')
    
#    print(np.max(block))
    
    if np.max(block) == 0:
        ske = np.zeros((1,1,nz,ny,nx))
        return ske
    else:
        pic[0,0] = block/np.max(block)
        img =torch.from_numpy(pic)
#        print(torch.max(img))
        
        #  predict
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
#        print(ske.shape)
        return ske[0,0]

if __name__ == "__main__":
    """    porcess the huge pic      """
    path = r'/home/wx/mycode/3D recover/predict_huge_pic/100108c3.tif'
    I = tifffile.imread(path)
    nZ,nY,nX = I.shape
    print("the huge_pic size:",nZ,nY,nX)
    

    pout = np.zeros_like(I)
    rz,ry,rx = 32,64,64
    
    #padding the Imgae
#    I_new = np.zeros((rz*round(nZ/rz), ry*round(nY/ry), rx*round(nX/rx)))
#    print(I_new.shape)

    np.pad(I,((0,nZ-rz),(0,nY-ry),(0,nX-rx)),'constant') 
    nZ,nY,nX = I.shape
    if nZ <rz or nY < ry or nX < rx :
        print("it is short of size")
        sys.exit(0)
        
    block =np.zeros((rz,ry,rx))
#    break
    for k in tqdm(range(0, nZ-rz,rz)):
        for j in range(0,nY-ry,ry):
            for i in range(0,nX-rx,rx):
#                print(k,j,i)
                block = I[k:k+rz, j:j+ry, i:i+rx]
                 
#                tifffile.imshow(block)
#                break 
#                print(np.max(block))

#                print(np.max(a))
                pout[k:k+rz, j:j+ry, i:i+rx] = predict_block(block)
    tifffile.imsave('ske.tif',pout)