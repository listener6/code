import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from WaveRNN import CustomRNN

epoch = 300

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

trained_model = CustomRNN(pml_coeff,source_function=s_t,varray_init=initModel,source_position=source_position,pml_width=pml_width,pml_decay=pml_decay)

# 加载预训练的模型
model_path = f"./model_save/{epoch}epoch.pth"
trained_model.load_state_dict(torch.load(model_path))

result = trained_model.varray.detach().cpu().numpy()
#显示结果
plt.imshow(result,cmap='jet',origin='upper',aspect='auto')
plt.colorbar(label='Velocity m/s')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title(f'{epoch}  result')
plt.show()

#保存结果
filename = f"./result/model_1/{epoch}epoch.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
np.savetxt(filename, result, delimiter="\t",fmt='%.9f')

ssss
