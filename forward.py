import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from WaveRNN import CustomRNN
import WaveRNN as WR

#模型名
local='model_2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建数据
num_timesteps = 1000   #时间点
nx, ny = 100, 100

# 边界条件初始化
pml_width=100
pml_decay=1

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

FM = 20 #主频
dt = 0.0005 #时间间隔
t = dt * (np.arange(num_timesteps) - (int)(1.0 / FM / dt))
#TODO: 需要调整振幅
A=1E6    #子波振幅
s_t =  A *( 1 - 2 * math.pi * math.pi * FM * FM * t * t) * np.exp(-math.pi * math.pi * FM * FM * t * t)
s_t = torch.tensor(s_t, dtype=torch.float).to(device)  # Move s_t to GPU

#模型设置
initModel=np.zeros((100,100))
initModel[0:30,:]=3000
initModel[30:60,:]=3500
initModel[60:,:]=4000
initModel = torch.tensor(initModel,dtype=torch.float)

# initModel=WR.readData("../traindata/model_3/modeltrue.txt")

# filename = "../traindata/model_1/model.txt"
# os.makedirs(os.path.dirname(filename), exist_ok=True)
# np.savetxt(filename, initModel, delimiter="\t",fmt='%.9f')

# input = (0,0)  #震源位置

model = CustomRNN(source_function=s_t,varray_init=initModel,pml_width=pml_width,pml_decay=pml_decay)
model = model.to(device)

#批量生成11炮炮集数据
for i in range(0,11):
    input=(0,i*10)
    result=torch.zeros(num_timesteps,ny)
    p1 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(device)
    p2 = torch.zeros(nx+2*pml_width, ny+2*pml_width).to(device)
    for t in range(0,num_timesteps):

        p3,output=model(input,t,p1,p2)
        result[t,:]=output
        p1=p2.to(device)
        p2=p3.to(device)

    # #查看波场
    # plt.imshow(p3.detach().numpy(),cmap='jet',origin='upper',aspect='auto')
    #查看炮集
    plt.imshow(result.detach().numpy(),cmap='jet',origin='upper',aspect='auto')
    plt.colorbar(label='')
    plt.xlabel('x (m)')
    plt.ylabel('t(0.0005s)')
    plt.title('Final result')
    plt.show()
    filename = f"../traindata/{local}/target_{i}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, result.detach().numpy(), delimiter="\t",fmt='%.9f')





    