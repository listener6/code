#单炮运行版本，均可运行
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter
import os
#定义MyDataset读取数据

#保存loss和model的文件位置
local='model_2'

def readData(file_path):
    path = file_path # 输入    
    # Load input txt files as channels
    
    with open(path, 'r') as f:
        lines = f.readlines()

    pattern = r'-?\d+\.?\d*'
    data = [[float(num) for num in re.findall(pattern, line)] for line in lines]
    # Convert input data to PyTorch tensor
    data = np.array(data)
    data = torch.from_numpy(data).float()
    

    return data


 #计算当前速度模型的梯度量
def getDeltaV(varray):
    nx,ny=varray.shape

    dvx = torch.zeros(nx,ny)
    dvz = torch.zeros(nx,ny)

    # 计算x方向梯度
    for i in range(1, 99):
        for j in range(100):
            dvx[i, j] = (varray[i+1, j] - varray[i-1, j]) / 2

    # 计算y方向梯度
    for i in range(100):
        for j in range(1, 99):
            dvz[i, j] = (varray[i, j+1] - varray[i, j-1]) / 2
    #归一化梯度矩阵
    if(torch.max(dvx) == torch.min(dvx)):
        dvx=dvx
    else:
        dvx = (dvx - torch.min(dvx)) / (torch.max(dvx) - torch.min(dvx))
    
    if(torch.max(dvz) == torch.min(dvz)):
        dvz=dvz
    else:
        dvz = (dvz - torch.min(dvz)) / (torch.max(dvz) - torch.min(dvz))

    dvx=dvx.to(device)
    dvz=dvz.to(device)

    return dvx, dvz

#设置边界条件


class CustomRNN(nn.Module):

    def __init__(self, nx=100, ny=100, dt=0.0005,nt=1000, dx=10, pml_width=40, pml_decay=1.0,
                 source_function=None, varray_init=None):
        super(CustomRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pml_width = pml_width
        self.pml_decay = pml_decay
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = dx
        self.nt=nt

        self.p1 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(self.device)
        self.p2 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(self.device)

        if varray_init is not None:
            self.varray = nn.Parameter(varray_init)
        else:
            self.varray = nn.Parameter(torch.ones(nx, ny))

        if source_function is None:
            self.source_function = torch.zeros(500)
        else:
            self.source_function = source_function


        

    def extend_with_pml(self,velocity_model, pml_thickness):
        ny, nx = velocity_model.shape

        # 创建一个具有扩展尺寸的新速度模型张量，并将其初始值设为 0
        extended_velocity_model = torch.zeros((ny + 2 * pml_thickness, nx + 2 * pml_thickness))

        # 将原始速度模型复制到扩展张量的中心
        extended_velocity_model[pml_thickness:-pml_thickness, pml_thickness:-pml_thickness] = velocity_model

        # 复制边界速度值以填充 PML 层
        extended_velocity_model[:pml_thickness, pml_thickness:-pml_thickness] = velocity_model[0, :].unsqueeze(0).expand(pml_thickness, nx)
        extended_velocity_model[-pml_thickness:, pml_thickness:-pml_thickness] = velocity_model[-1, :].unsqueeze(0).expand(pml_thickness, nx)
        extended_velocity_model[pml_thickness:-pml_thickness, :pml_thickness] = velocity_model[:, 0].unsqueeze(1).expand(ny, pml_thickness)
        extended_velocity_model[pml_thickness:-pml_thickness, -pml_thickness:] = velocity_model[:, -1].unsqueeze(1).expand(ny, pml_thickness)

        # 复制角点
        extended_velocity_model[:pml_thickness, :pml_thickness] = velocity_model[0, 0]
        extended_velocity_model[-pml_thickness:, :pml_thickness] = velocity_model[-1, 0]
        extended_velocity_model[:pml_thickness, -pml_thickness:] = velocity_model[0, -1]
        extended_velocity_model[-pml_thickness:, -pml_thickness:] = velocity_model[-1, -1]

        extended_velocity_model=extended_velocity_model.to(self.device)
        return extended_velocity_model
    
   



    # #添加PML版本
    # #修改forward，炮位置x作为网络参数传入，炮集作为输出
    # def forward(self,  x):

    #     x_s, y_s = x
    #     pml_varray=self.extend_with_pml(self.varray,self.pml_width)  #速度需要按照PML进行扩充
    #     alpha = (pml_varray ** 2 )*( self.dt ** 2 )   #alpha参数在一轮中，速度不变，参数也不变
        
    #     #稳定性判断
    #     DIFF_COEFF = np.array([0, 0.1261138E+1, -0.1297359E+0, 0.3989181E-1, -0.1590804E-1, 0.6780797E-2, -0.2804773E-2, 0.1034639E-2,-0.2505054E-3])
    #     limit = pml_varray.max() * self.dt * np.sqrt(1.0/self.dx/self.dx+1/self.dx/self.dx)*np.sum(np.fabs(DIFF_COEFF))
    #     if(limit > 1):
    #         print("limit = ")
    #         print(limit)
    #         print("不满足稳定性条件.")
    #         exit(0)

    #     #初始波场p1,p2
    #     p1 = self.p1
    #     p2 = self.p2
    #     gather = torch.zeros((self.nt,self.ny))
    #     for t in range(0, self.nt):
            

    #         #边界8个网格不算，衰减后可看作0
    #         dpdx = (p2[:, 9:-7] - 2 * p2[:, 8:-8] + p2[:, 7:-9]) / (self.dx ** 2)
            
    #         dpdy = (p2[9:-7, :] - 2 * p2[8:-8, :] + p2[7:-9, :]) / (self.dx ** 2)
            
    #         rhs = 2 * p2[8:-8, 8:-8] - p1[8:-8, 8:-8] + alpha[8:-8, 8:-8] * (dpdx[8:-8, :] + dpdy[:, 8:-8])
            
    #         p = torch.zeros_like(self.p1)
    #         p[8:-8,8:-8]=rhs
            
    #         #添加震源，p是扩充后波场
    #         p[x_s+self.pml_width, y_s+self.pml_width] += self.source_function[t]  * (self.dt ** 2)
            
            
    #         #res是所需大小波场
    #         res=p[self.pml_width:-self.pml_width,self.pml_width:-self.pml_width]
    #         gather[t,:]=res[:,50]

            
    #         p1=p2
            
    #         p2=p


    #     return p,gather


    #添加PML版本
    #修改forward，炮位置x、时间t作为网络参数传入，当前时刻波场和炮集
    def forward(self,  x, t, p1, p2):

        x_s, y_s = x
        pml_varray=self.extend_with_pml(self.varray,self.pml_width)  #速度需要按照PML进行扩充
        alpha = (pml_varray ** 2 )*( self.dt ** 2 )   #alpha参数在一轮中，速度不变，参数也不变
        
        #稳定性判断
        DIFF_COEFF = np.array([0, 0.1261138E+1, -0.1297359E+0, 0.3989181E-1, -0.1590804E-1, 0.6780797E-2, -0.2804773E-2, 0.1034639E-2,-0.2505054E-3])
        limit = pml_varray.max() * self.dt * np.sqrt(1.0/self.dx/self.dx+1/self.dx/self.dx)*np.sum(np.fabs(DIFF_COEFF))
        if(limit > 1):
            print("limit = ")
            print(limit)
            print("不满足稳定性条件.")
            exit(0)

        #初始波场p1,p2
 
        #边界8个网格不算，衰减后可看作0
        dpdx = (p2[:, 9:-7] - 2 * p2[:, 8:-8] + p2[:, 7:-9]) / (self.dx ** 2)
        
        dpdy = (p2[9:-7, :] - 2 * p2[8:-8, :] + p2[7:-9, :]) / (self.dx ** 2)
        
        rhs = 2 * p2[8:-8, 8:-8] - p1[8:-8, 8:-8] + alpha[8:-8, 8:-8] * (dpdx[8:-8, :] + dpdy[:, 8:-8])
        
        p = torch.zeros_like(self.p1)
        p[8:-8,8:-8]=rhs
        
        #添加震源，p是扩充后波场
        p[x_s+self.pml_width, y_s+self.pml_width] += self.source_function[t]  * (self.dt ** 2)
        
        
        #res是所需大小波场
        res=p[self.pml_width:-self.pml_width,self.pml_width:-self.pml_width]
        


        return p,res[:,50]

    

if __name__=="__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据大小初始化
    num_timesteps = 1000   #时间点
    nx, ny = 100, 100

    # 边界条件初始化
    pml_width=100
    pml_decay=1

    #读入target
    targets=torch.zeros(11,num_timesteps,ny)
    for i in range(0,11):
        targets[i]=readData(f"../traindata/model_2/target_{i}.txt")

    pml_coeff=torch.ones((nx+2*pml_width, ny+2*pml_width), dtype=torch.float32)
    # for i in range(pml_width):
    #     # decay = ((pml_width - i) / self.pml_width) ** self.pml_decay
    #     # decay =(1 - ((pml_width - i) / pml_width) ** 4)
    #     decay = pow(pml_decay, pml_width - i)
    #     pml_coeff[i, :] *= decay
    #     pml_coeff[-i - 1, :] *= decay
    #     pml_coeff[:, i] *= decay
    #     pml_coeff[:, -i - 1] *= decay

    pml_coeff=pml_coeff.to(device)

    # 子波初始化
    source_position = (0,50)  #震源位置
    FM = 20 #主频
    dt = 0.0005 #时间间隔
    t = dt * (np.arange(num_timesteps) - (int)(1.0 / FM / dt))
    #TODO: 需要调整振幅
    A=1E6    #子波振幅
    s_t =  A *( 1 - 2 * math.pi * math.pi * FM * FM * t * t) * np.exp(-math.pi * math.pi * FM * FM * t * t)
    s_t = torch.tensor(s_t, dtype=torch.float).to(device)  # Move s_t to GPU


    #真实速度模型
    vTrue=np.zeros((100,100))
    vTrue[0:30,:]=3000
    vTrue[30:60,:]=3500
    vTrue[60:,:]=4000
    vTrue = torch.tensor(vTrue,dtype=torch.float).to(device)
    target_dvx,target_dvz=getDeltaV(vTrue)

    # 初始速度模型
    initModel=np.zeros((100,100))
    initModel[0:30,:]=3000
    initModel[30:60,:]=3400
    initModel[60:,:]=3900
    initModel = torch.tensor(initModel,dtype=torch.float).to(device)

    # 模型初始化
    model = CustomRNN(source_function=s_t,varray_init=initModel,pml_width=pml_width,pml_decay=pml_decay)
    model = model.to(device)

    # 设定损失函数和优化器
    criterion = nn.MSELoss()
    criterion=criterion.to(device)
    writer = SummaryWriter(f"../loss/{local}")
    #TODO:损失太小了，需要调整学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

    # 训练循环
    num_epochs = 8000
    i=0
    # input=(0,50)
    for epoch in range(num_epochs):

        loss = 0.0
        #计算速度结构损失
        dvx,dvz=getDeltaV(model.varray)
        loss+=criterion(dvx,target_dvx)
        loss+=criterion(dvz,target_dvz)
        #波场损失和速度结构损失作为共同损失
        for j in range(0,11):
            
            input=(0,j*10)
            p1 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(device)
            p2 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(device)
            loss_t=0
            for t in range(0,num_timesteps):

                p3,output=model(input,t,p1,p2)
                output=output.to(device)
                target=targets[j,t,:].to(model.device)
                loss_t+=criterion(output, target)

                p1=p2.to(device)
                p2=p3.to(device)

            loss+=loss_t
        

        writer.add_scalar('training loss', loss.item(), i)
        i=i+1
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        optimizer.step ()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"../model_save/{local}/model_f_{epoch + 1}.pth")
            print("model saved")

    # # 训练结束后输出参数varray
    # trained_varray = model.varray.detach().cpu().numpy()

    # print("Trained varray:")
    # print(trained_varray)


# #自定义优化器
# class CustomOptimizer(optim.Optimizer):
#     def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0):
#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
#         super(CustomOptimizer, self).__init__(params, defaults)

#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             lr = group['lr']
#             momentum = group['momentum']
#             weight_decay = group['weight_decay']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data

#                 if p[:30, :].numel() > 0:
#                     p[:30, :].data.add_(-lr * torch.mean(grad[:30, :], dim=0))

#                 if p[30:60, :].numel() > 0:
#                     p[30:60, :].data.add_(-lr * torch.mean(grad[30:60, :], dim=0))

#                 if p[60:, :].numel() > 0:
#                     p[60:, :].data.add_(-lr * torch.mean(grad[60:, :], dim=0))

#                 if weight_decay != 0:
#                     p.data.add_(-weight_decay * lr, p.data)

#         return loss


