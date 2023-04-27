import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from WaveRNN import CustomRNN




# 创建数据
num_timesteps = 1000   #时间点
nx, ny = 100, 100

#PML设置
pml_width=200
pml_decay=1
pml_coeff=torch.ones((nx+2*pml_width, ny+2*pml_width), dtype=torch.float32)

#震源设置
source_position = (0,50)  #震源位置
FM = 20 #主频
dt = 0.0005 #时间间隔
t = dt * (np.arange(num_timesteps) - (int)(1.0 / FM / dt))
#TODO: 需要调整振幅
A=1E6    #子波振幅
s_t =  A *( 1 - 2 * math.pi * math.pi * FM * FM * t * t) * np.exp(-math.pi * math.pi * FM * FM * t * t)

#模型设置
initModel=np.zeros((100,100))
initModel[0:30,:]=3000
initModel[30:60,:]=3500
initModel[60:,:]=4000
initModel = torch.tensor(initModel,dtype=torch.float)

model = CustomRNN(pml_coeff,source_function=s_t,varray_init=initModel,source_position=source_position,pml_width=pml_width,pml_decay=pml_decay)
p1 = torch.zeros(nx+2*pml_width, ny+2*pml_width)
p2 = torch.zeros(nx+2*pml_width, ny+2*pml_width)
result=np.zeros((num_timesteps,100))
for t in range(0, num_timesteps):
    # 前向传播
    p3,output = model(p1, p2,t)
    result[t,:] = output.detach().numpy() #炮集
    # result = p3.detach().numpy()  # 看波场快照

    p1=p2
    p2=p3

res=result[50:,:]
# plt.imshow(result,cmap='jet',origin='upper',aspect='auto')
# plt.colorbar(label='')
# plt.xlabel('x (m)')
# plt.ylabel('t(0.0005s)')
# plt.title('Final result')
# plt.show()


filename = "../traindata/model_1/target.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
np.savetxt(filename, result, delimiter="\t",fmt='%.9f')


    